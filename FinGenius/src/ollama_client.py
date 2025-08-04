"""Ollama客户端包装器，确保base_url不被修改"""
import httpx
import json
from typing import Any, Dict
from src.logger import logger


class OllamaAsyncOpenAI:
    """
    专门为Ollama设计的简化客户端
    直接使用httpx发送请求，避免OpenAI SDK的URL处理问题
    """
    
    def __init__(self, base_url: str, api_key: str = "ollama", **kwargs):
        # 记录原始base_url，移除末尾的斜杠
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        
        logger.info(f"OllamaAsyncOpenAI initializing with base_url: {self.base_url}")
        
        # 创建HTTP客户端
        self._http_client = httpx.AsyncClient(
            timeout=kwargs.get('timeout', 300.0),
            follow_redirects=False,  # 禁用重定向
        )
        
        logger.info(f"OllamaAsyncOpenAI initialized successfully")
        logger.info(f"  - Base URL: {self.base_url}")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._http_client.aclose()
    
    @property
    def chat(self):
        """返回chat接口的包装器"""
        return ChatCompletions(self)


class ChatCompletions:
    """Chat completions接口包装器"""
    
    def __init__(self, client: OllamaAsyncOpenAI):
        self.client = client
    
    @property
    def completions(self):
        """返回completions接口"""
        return Completions(self.client)


class Completions:
    """Completions接口实现"""
    
    def __init__(self, client: OllamaAsyncOpenAI):
        self.client = client
    
    async def create(self, **kwargs) -> Any:
        """
        创建chat completion请求
        """
        logger.info(f"Ollama API call:")
        logger.info(f"  - Target URL: {self.client.base_url}/chat/completions")
        logger.info(f"  - Model: {kwargs.get('model', 'Unknown')}")
        logger.info(f"  - Stream: {kwargs.get('stream', False)}")
        
        # 构建请求URL
        url = f"{self.client.base_url}/chat/completions"
        
        # 设置请求头
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.client.api_key}',
        }
        
        try:
            # 发送请求
            response = await self.client._http_client.post(
                url,
                headers=headers,
                json=kwargs
            )
            
            logger.info(f"Ollama API call successful - Status: {response.status_code}")
            
            # 处理流式响应
            if kwargs.get('stream', False):
                return self._handle_stream_response(response)
            else:
                # 非流式响应
                response.raise_for_status()
                result = response.json()
                return self._create_completion_object(result)
                
        except Exception as e:
            logger.error(f"Ollama API call failed: {str(e)}")
            logger.error(f"  - Exception type: {type(e).__name__}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"  - Response status: {e.response.status_code}")
                logger.error(f"  - Response text: {e.response.text}")
            raise
    
    def _create_completion_object(self, data: Dict) -> Any:
        """创建兼容OpenAI格式的响应对象"""
        # 创建一个简单的响应对象，兼容OpenAI的接口
        class CompletionResponse:
            def __init__(self, data):
                self.choices = []
                self.usage = None
                
                if 'choices' in data:
                    for choice_data in data['choices']:
                        choice = type('Choice', (), {})()
                        if 'message' in choice_data:
                            choice.message = type('Message', (), choice_data['message'])()
                        self.choices.append(choice)
                
                if 'usage' in data:
                    self.usage = type('Usage', (), data['usage'])()
        
        return CompletionResponse(data)
    
    async def _handle_stream_response(self, response):
        """处理流式响应"""
        async for line in response.aiter_lines():
            if line.startswith('data: '):
                data_str = line[6:]
                if data_str.strip() == '[DONE]':
                    break
                try:
                    data = json.loads(data_str)
                    yield self._create_stream_chunk(data)
                except json.JSONDecodeError:
                    continue
    
    def _create_stream_chunk(self, data: Dict) -> Any:
        """创建流式响应块"""
        class StreamChunk:
            def __init__(self, data):
                self.choices = []
                if 'choices' in data:
                    for choice_data in data['choices']:
                        choice = type('Choice', (), {})()
                        if 'delta' in choice_data:
                            choice.delta = type('Delta', (), choice_data['delta'])()
                        self.choices.append(choice)
        
        return StreamChunk(data)
