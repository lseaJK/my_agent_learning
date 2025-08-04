```
classDiagram
    %%==========================
    %% 1. Agent 层次结构
    %%==========================
    class BaseAgent {
        <<abstract>>
        - name: str
        - description: Optional[str]
        - system_prompt: Optional[str]
        - next_step_prompt: Optional[str]
        - llm: LLM
        - memory: Memory
        - state: AgentState
        - max_steps: int
        - current_step: int
        + run(request: Optional[str]) str
        + step() str <<abstract>>
        + update_memory(role, content, **kwargs) None
        + reset_execution_state() None
    }
    class ReActAgent {
        <<abstract>>
        + think() bool <<abstract>>
        + act() str <<abstract>>
        + step() str
    }
    class ToolCallAgent {
        - available_tools: ToolCollection
        - tool_choices: ToolChoice
        - tool_calls: List~ToolCall~
        + think() bool
        + act() str
        + execute_tool(command: ToolCall) str
        + cleanup() None
    }
    class MCPAgent {
        - mcp_clients: MCPClients
        - tool_schemas: Dict~str, dict~
        - connected_servers: Dict~str, str~
        - initialized: bool
        + create(...) MCPAgent
        + initialize_mcp_servers() None
        + connect_mcp_server(url, server_id) None
        + initialize(...) None
        + _refresh_tools() Tuple~List~str~,List~str~~
        + cleanup() None
    }
    class SentimentAgent {
        <<extends MCPAgent>>
    }
    class RiskControlAgent {
        <<extends MCPAgent>>
    }
    class HotMoneyAgent {
        <<extends MCPAgent>>
    }
    class TechnicalAnalysisAgent {
        <<extends MCPAgent>>
    }
    class ReportAgent {
        <<extends MCPAgent>>
    }

    %% 继承关系
    BaseAgent <|-- ReActAgent
    ReActAgent <|-- ToolCallAgent
    ToolCallAgent <|-- MCPAgent
    MCPAgent <|-- SentimentAgent
    MCPAgent <|-- RiskControlAgent
    MCPAgent <|-- HotMoneyAgent
    MCPAgent <|-- TechnicalAnalysisAgent
    MCPAgent <|-- ReportAgent

    %% 组合关系
    BaseAgent *-- LLM               : llm
    BaseAgent *-- Memory            : memory
    MCPAgent *-- MCPClients         : mcp_clients

    %%==========================
    %% 2. 环境（Environment）架构
    %%==========================
    class BaseEnvironment {
        <<abstract>>
        - name: str
        - description: str
        - agents: Dict~str, BaseAgent~
        - max_steps: int
        + create(...) BaseEnvironment
        + register_agent(agent: BaseAgent) None
        + run(...) Dict~str,Any~ <<abstract>>
        + cleanup() None
    }
    class ResearchEnvironment {
        - analysis_mapping: Dict~str,str~
        - results: Dict~str,Any~
        + initialize() None
        + run(stock_code: str) Dict~str,Any~
        + cleanup() None
    }
    class BattleState {
        - active_agents: Dict~str,str~
        - voted_agents: Dict~str,str~
        - terminated_agents: Dict~str,bool~
        - battle_history: List~Dict~str,Any~~
        - vote_results: Dict~str,int~
        - battle_highlights: List~Dict~str,Any~~
        - battle_over: bool
        + add_event(type, agent_id, ...) Dict~str,Any~
        + record_vote(agent_id,vote) None
        + mark_terminated(agent_id,reason) None
    }
    class BattleEnvironment {
        - state: BattleState
        - tools: Dict~str,BaseTool~
        + initialize() None
        + register_agent(agent: BaseAgent) None
        + run(report: Dict~str,Any~) Dict~str,Any~
        + handle_speak(agent_id, content) ToolResult
        + handle_vote(agent_id, vote) ToolResult
        + cleanup() None
    }
    class EnvironmentFactory {
        + create_environment(env_type: EnvironmentType, agents, ...) BaseEnvironment
    }

    %% 继承与工厂
    BaseEnvironment <|-- ResearchEnvironment
    BaseEnvironment <|-- BattleEnvironment
    EnvironmentFactory ..> BaseEnvironment : creates
    %% 环境中包含 Agents 和 BattleState
    BaseEnvironment o-- BaseAgent      : agents
    BattleEnvironment *-- BattleState  : state

    %%==========================
    %% 3. 工具（Tool）抽象
    %%==========================
    class MCPClients {
        - sessions: Dict~str,ClientSession~
        - exit_stacks: Dict~str,AsyncExitStack~
        + connect_sse(url, server_id) None
        + connect_stdio(cmd, args, server_id) None
        + list_tools() ListToolsResult
        + disconnect(server_id) None
    }

    %%==========================
    %% 4. 支持类
    %%==========================
    class Memory {
        - messages: List~Message~
        + add_message(msg: Message) None
        + clear() None
    }
    class LLM {
        - model: str
        - max_tokens: int
        - temperature: float
        + ask(messages, system_msgs, ...) str
        + ask_tool(messages, tools, tool_choice, ...) Message
    }
```