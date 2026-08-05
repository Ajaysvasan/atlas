**Problem Statement**

Large Language Models (LLMs) have become increasingly capable of assisting users in software development, research, and technical problem solving. However, their ability to maintain long-term project-specific context remains limited. Existing conversational AI systems typically operate within a fixed context window, causing earlier discussions to be forgotten as conversations grow longer. This limitation forces users to repeatedly provide context, reducing productivity and increasing the likelihood of inconsistent or inaccurate responses.

Current Retrieval-Augmented Generation (RAG) systems partially address this issue by retrieving relevant information from external knowledge bases using vector similarity search. However, these systems primarily focus on document retrieval and do not maintain structured memory of ongoing project conversations. As a result, they lack the ability to preserve conversational continuity, organize knowledge hierarchically across projects, and efficiently retrieve relevant historical context.

Furthermore, many existing solutions depend on cloud-based vector databases and external AI services, making them unsuitable for users requiring local execution, privacy, and complete ownership of their project knowledge.

Therefore, there is a need for a local, project-aware memory system capable of organizing conversations, preserving long-term context, efficiently retrieving relevant information, and supporting scalable knowledge management without relying on external infrastructure.

**Solution Statement**

This project proposes a local Project-Aware Hierarchical Memory and Retrieval System that extends traditional Retrieval-Augmented Generation by introducing structured project-specific memory management. The system enables users to build persistent knowledge bases for multiple projects while maintaining conversational continuity through hierarchical memory organization.

##**Project components**
