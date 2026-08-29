# Project

**Problem Statement**

Large Language Models (LLMs) have become increasingly capable of assisting users in software development, research, and technical problem solving. However, their ability to maintain long-term project-specific context remains limited. Existing conversational AI systems typically operate within a fixed context window, causing earlier discussions to be forgotten as conversations grow longer. This limitation forces users to repeatedly provide context, reducing productivity and increasing the likelihood of inconsistent or inaccurate responses.

Current Retrieval-Augmented Generation (RAG) systems partially address this issue by retrieving relevant information from external knowledge bases using vector similarity search. However, these systems primarily focus on document retrieval and do not maintain structured memory of ongoing project conversations. As a result, they lack the ability to preserve conversational continuity, organize knowledge hierarchically across projects, and efficiently retrieve relevant historical context.

Furthermore, many existing solutions depend on cloud-based vector databases and external AI services, making them unsuitable for users requiring local execution, privacy, and complete ownership of their project knowledge.

Therefore, there is a need for a local, project-aware memory system capable of organizing conversations, preserving long-term context, efficiently retrieving relevant information, and supporting scalable knowledge management without relying on external infrastructure.

**Solution Statement**

This project proposes a local Project-Aware Hierarchical Memory and Retrieval System that extends traditional Retrieval-Augmented Generation by introducing
structured project-specific memory management. The system enables users to build persistent knowledge bases for multiple projects while maintaining conversational
continuity through hierarchical memory organization.

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

###### 1. Conversation summary , where each chunk of the summary generated is embedded and stored

###### 2. cummulative summary , where the entire summary with out any summary gets embedded and stored

This layer is responsible for generating appropirate answers and storing the context for that conversation and project.
This layer mainly exists to reduce the amount of hallucations and also to solve the problem of context rotting. We will see more about them in the working section
This layer also uses CBR , where we only store the vector ids here , and when we get a query we use the vector IDs to get the vectors from the postgresql
for CBR

### 3. Query Decomposition layer

This layer decomposes the query into two major categories

#### 1. mode 1 query (development / plan query)-> where the user query is sort of like this "Help me to develop " or any queries similar to that

#### 2. mode 2 query (doubt query)-> where the user query is sort of like "What does this do? " or any queries similar to that

This layer helps in identifing the query the helps the model to make plans accordingly , for instance
a simple query
`Help me to build a physics engine in C `
is not as simple as it looks , it contains complex things like

1. Mathematics
2. physics equations
3. coding
4. Creating appropirate data structures
   and many more things

So by decomposing the query to understand the hidden dependency and re-structuring them would help the model to plan efficiently
also if we get a query like
`Why do we use vectors instead of array here`
then the model should not spend much time in it , rather it can just see either full conversation or appropirate summary to answer that rather than thinking heavily

### 4. The planner critique layer

#### 1. Planner layer

The planner is the one that takes the mode 1 queries , and first plan the solution first.
There will be 4 worker planners and 1 master planner

Each of the 4 worker planner takes up one hypothesis from the query and work on it
The master query is the one that finalizes the overall solution by stritching all the solutions provided by each hypothesis

#### 2. Critique Layer:

The critique layer takes the solution from the planner and find the loop holes , flaws , checks for fessiablity , checks for possible things that can go wrong ,
checks if the solution is bound to the scope and returns that to the planner

the planner then takes up that feedback from the critique and tries to find a solution with the existing solutiion or come up with new solution or together

### 4. Retrieval layer

The retrieval layer is responsible for retrieving the chunks that is needed by the planner to make the solution

### 5. Generation layer

This where the LLM lives , once the solution is validated by the critique , it is passed to the generation layer.
Here the role of the LLM is generate a response which is human like along with the solution without chaning it.
This is for human readability

### 6. Knowledge accqusation

This layer is there to check if the system has enough knowledge to address the given query is yes , then we proceed with the retrieval ,
other wise the knowledge accqusation layer goes to the internet , crawls over it and take the neccessary data and add it to the vector DB
once the neccessary data is obtained that's when the retrieval takes place
