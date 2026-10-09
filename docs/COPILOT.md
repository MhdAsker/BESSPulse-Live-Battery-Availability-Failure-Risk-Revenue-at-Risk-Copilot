# Copilot grounding policy

Operational tools are authoritative for current telemetry, availability, delivery risk, alerts, market context, Revenue-at-Risk, and component status. The approved documentation index is authoritative only for definitions, methodology, runbooks, interpretation, and recorded model limitations. Documentation defaults never override live asset configuration.

`search_knowledge_base(query, top_k)` accepts no filesystem path and returns chunks with stable source/section citations and relevance scores. Current-state-only questions are refused when operational tools are unavailable rather than answered from RAG. Mixed questions require both tool evidence and RAG; because Prompt 13 agent orchestration is absent in this repository, the compatibility endpoint labels mixed answers partial and explicitly withholds current-state claims.

Set `RAG_ENABLED=1` only after an index has been created by the explicit RAG job. The default remains disabled. Gemini synthesis is optional and is not implemented by the current compatibility service.
