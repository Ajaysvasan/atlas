# Project

**Problem Statement**

Large Language Models (LLMs) have become increasingly capable of assisting users in software development, research, and technical problem solving. However, their ability to maintain long-term project-specific context remains limited. Existing conversational AI systems typically operate within a fixed context window, causing earlier discussions to be forgotten as conversations grow longer. This limitation forces users to repeatedly provide context, reducing productivity and increasing the likelihood of inconsistent or inaccurate responses.

Current Retrieval-Augmented Generation (RAG) systems partially address this issue by retrieving relevant information from external knowledge bases using vector similarity search. However, these systems primarily focus on document retrieval and do not maintain structured memory of ongoing project conversations. As a result, they lack the ability to preserve conversational continuity, organize knowledge hierarchically across projects, and efficiently retrieve relevant historical context.

Furthermore, many existing solutions depend on cloud-based vector databases and external AI services, making them unsuitable for users requiring local execution, privacy, and complete ownership of their project knowledge.

Therefore, there is a need for a local, project-aware memory system capable of organizing conversations, preserving long-term context, efficiently retrieving relevant information, and supporting scalable knowledge management without relying on external infrastructure.

**Solution Statement**

This project proposes a local Project-Aware Hierarchical Memory and Retrieval System that extends traditional Retrieval-Augmented Generation by introducing structured project-specific memory management. The system enables users to build persistent knowledge bases for multiple projects while maintaining conversational continuity through hierarchical memory organization.

## Project components

The project is decomposed into the following major components,

### 1. Data layer :

This layer is responsible for extracting data from the dataset i.e from files with the following extension , (.md , .txt , .docs , .pdf)
and preprocess them , chunk them , store the chunks in sql , embedded them and store the embeddings in postgresql and diskANN
here we use diskann as our vector DB
The responsibility of the Data layer is to store and manage data and embedding nothing more that.

### 2. Memory layer

The memory layer is the heart of this project.
It is sub divided into the following components

#### 1. Topic pool :

The topic pool is the one that maintains all the topics. The topic can be AI , web development , backend engineering , Thermodynamics extracting
It is not only scoped to coding but rather as a general one.
This layer is the one that manages all the topics and decides which topic should the given query belongs to and pass that to query to the project manager

#### 2. Project pool :

The project pool contains the project manager which manages all the project which belongs to a particular topic , this is the layer which decides whether we need a new project of that topic is to be created or
if the query belongs to already existing project. If the query belongs to an existing project then it passes the query to the conversation manager

#### 3. Conversation pool :

The conversation pool stores the conversation in two ways

##### 1. Full conversation -> The entire conversation that is from what is the user query , what did the llm retrieve , they are stored as whole.

##### 2. Conversation summary :

        this is futher sub divided into 2 buckets
            1. Conversation summary , where each chunk of the summary generated is embedded and stored
            2. cummulative summary , where the entire summary with out any summary gets embedded and stored

This layer is responsible for generating appropirate answers and storing the context for that conversation and project.
This layer mainly exists to prevent hallucinations and also to solve the problem of context rotting. We will see more about them in the working section
