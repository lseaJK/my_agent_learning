# FinGenius 核心函数级调用流程详解

本文档旨在深入 `main.py` 的执行流程，并追踪其与 `src` 目录中关键类和核心函数的交互细节。

#### **顶层编排: `main.py`**

程序的入口是 `async def main()`，其核心逻辑是实例化并调用 `EnhancedFinGeniusAnalyzer`。

```
[CMD] python main.py <stock_code>
  |
  v
asyncio.run(main())
  |
  v
  1. args = parser.parse_args()  // 解析命令行参数
  |
  v
  2. analyzer = EnhancedFinGeniusAnalyzer() // 创建主控制器
  |
  v
  3. results = await analyzer.analyze_stock(args.stock_code, ...) // **启动核心分析**
  |
  v
  4. display_results(results, ...) // 显示结果
  |
  v
  5. await announce_result_with_tts(results) // (可选) TTS播报
```

#### **核心分析: `EnhancedFinGeniusAnalyzer.analyze_stock()`**

这是整个工作流的“大脑”，它严格按照顺序编排了三个主要阶段。

```
analyze_stock(stock_code, ...):
  |
  +--> visualizer.show_section_header("研究阶段", "🔍")
  |
  v
  1. research_results = await self._run_research_phase(stock_code, ...) // **进入研究阶段**
  |
  +--> visualizer.show_section_header("专家辩论阶段", "⚔��")
  |
  v
  2. battle_results = await self._run_battle_phase(research_results, ...) // **进入辩论阶段**
  |
  v
  3. await self._generate_reports(stock_code, research_results, battle_results) // **进入报告阶段**
  |
  v
  4. return self._prepare_final_results(...) // 准备并返回最终结果
```

---

### **第一阶段：研究 (Function-Level Deep Dive)**

**Orchestrator: `_run_research_phase(stock_code, ...)`**

1.  `research_env = await ResearchEnvironment.create(...)`
    *   **内部调用**: `ResearchEnvironment.__init__()`
    *   **动作**:
        *   初始化一个 `LLM` 客户端实例 (如 `src.llm.SiliconFlowLLM`)。
        *   循环遍历一个预定义的 Agent 类列表（如 `SentimentAgent`, `RiskControlAgent` 等）。
        *   对每个 Agent 类，调用 `await AgentClass.create(llm_client, ...)` 来创建 Agent 实例。
        *   将创建的 Agent 实例存储在 `research_env.agents` 字典中。

2.  `await AgentClass.create(llm_client, ...)` (以 `SentimentAgent` 为例)
    *   **内部调用**: `SentimentAgent.__init__()` -> `BaseAgent.__init__()`
    *   **动作**:
        *   加载自身的 Prompt (从 `src.prompt.sentiment.PROMPT`)。
        *   实例化 `ToolCollection` (`src.tool.tool_collection.ToolCollection`)。
        *   `ToolCollection` 会实例化该 Agent 被授权的所有工具 (如 `src.tool.web_search.WebSearchTool`) 并存入 `self.tools` 字典。
        *   返回 `SentimentAgent` 的实例。

3.  `results = await research_env.run(stock_code)`
    *   **动作**: 这是研究阶段的核心循环。
    *   **内部调用链**:
        ```
        research_env.run(stock_code):
          |
          v
          for agent in self.agents.values(): // 遍历所有研究员
            |
            v
            // **核心任务执行**
            agent_result = await agent.run(task=f"分析 {stock_code} 的{agent.name}")
            |
            v
            results[agent.name] = agent_result
          |
          v
          return results
        ```

4.  `agent.run(task)`
    *   **动作**: 这是单个 Agent 执行其分析任务的 ReAct (Reason-Act) 循环。
    *   **内部调用链 (简化版)**:
        ```
        agent.run(task):
          |
          v
          // 1. 第一次调用 LLM (思考)
          llm_response = await self.llm.chat_completion(prompt=task, tools=self.available_tools)
          |
          +-- if llm_response contains tool_call:
          |     |
          |     v
          |     // 2. 执行工具
          |     tool_result = await self.available_tools.execute(
          |         name=tool_call.name,
          |         tool_input=tool_call.arguments
          |     )
          |     |
          |     +--> ToolCollection.execute()
          |          |
          |          +--> SpecificTool.execute() // e.g., WebSearchTool.execute()
          |
          |     v
          |     // 3. 将工具结果再次输入 LLM
          |     llm_response = await self.llm.chat_completion(prompt=..., tool_result=tool_result)
          |     // ... 这个循环会持续，直到 LLM 给出最终答案或达到 max_steps
          |
          +-- else (llm_response is final answer):
                |
                v
                return llm_response.content
        ```

---

### **第二阶段：辩论 (Function-Level Deep Dive)**

**Orchestrator: `_run_battle_phase(research_results, ...)`**

1.  `battle_env = await BattleEnvironment.create(...)`
    *   **动作**: 创建辩论环境，初始化状态，包括空的辩论历史 `debate_history`、轮次计数器等。

2.  `battle_env.register_agent(agent)`
    *   **动作**: 将所有专家 Agent 注册到辩论环境中。

3.  `results = await battle_env.run(research_results)`
    *   **动作**: 启动并管理整个辩论和投票过程。
    *   **内部调用链**:
        ```
        battle_env.run(research_results):
          |
          v
          // 1. 辩���循环
          for round_num in range(self.debate_rounds):
            |
            v
            for agent in self.agent_order: // 按顺序发言
              |
              v
              // a. 准备给 Agent 的上下文 (包含 research_results 和 debate_history)
              context = self._prepare_agent_context(agent, research_results, self.debate_history)
              |
              v
              // b. 让 Agent 发言
              speech = await agent.speak(context)
              |
              v
              // c. 广播消息 (main.py 中被增强以实现可视化)
              await self._broadcast_message(sender=agent.id, content=speech, event_type="speak")
              |   |
              |   +--> self.debate_history.append(...) // 记录发言
              |
              v
          // 2. 投票环节
          for agent in self.agent_order:
            |
            v
            // a. 准备投票上下文
            vote_context = self._prepare_voting_context(self.debate_history)
            |
            v
            // b. 让 Agent 投票
            vote = await agent.vote(vote_context)
            |
            v
            // c. 记录投票
            self.votes[agent.id] = vote
          |
          v
          // 3. 结果汇总
          final_decision, vote_count = self._tally_votes(self.votes)
          battle_highlights = await self._summarize_debate(self.debate_history) // 可能再次调用LLM
          |
          v
          return { "final_decision": ..., "vote_count": ..., "debate_history": ... }
        ```

---

### **第三阶段：报告生成 (Function-Level Deep Dive)**

**Orchestrator: `_generate_reports(...)`**

1.  `report_agent = await ReportAgent.create(...)`
    *   **动作**: 创建一个专门用于生成报告的 Agent，它被授权使用 `CreateHTMLTool`。

2.  `html_request = f"""..."""`
    *   **动作**: 构造一个巨大的、包含所有分析和辩论结果的字符串。这本质上是一个 **One-Shot Prompt**。

3.  `await report_agent.available_tools.execute(name="create_html", ...)`
    *   **内部调用**: `ToolCollection.execute()` -> `CreateHTMLTool.execute(request, output_path, data)`
    *   **`CreateHTMLTool.execute()` 内部调用链**:
        ```
        CreateHTMLTool.execute(request, output_path, data):
          |
          v
          // 1. 准备给 LLM 的 Prompt
          prompt = f"System: You are an expert web developer... User: {request} Data: {json.dumps(data)}"
          |
          v
          // 2. 调用 LLM 生成代码
          html_content = await self.llm.chat_completion(prompt=prompt)
          |
          v
          // 3. 写入文件
          with open(output_path, "w") as f:
              f.write(html_content)
        ```

4.  `report_manager.save_debate_report(debate_data, ...)`
    *   **内部调用**: `json.dump(debate_data, f, ...)`
    *   **动作**: 这是一个纯粹的文件 I/O 操作，不涉及 LLM。它将结构化的 Python 字典序列化为 JSON 并保存到磁盘。
