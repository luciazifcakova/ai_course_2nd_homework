# ai_course_2nd_homework
Local file search engine

# How to use:

1. ingest the dir in this github repo into knowledge outside of your project, use some model that can turn text into vector
2. use any llm model for the agent
3. make sure Knowledge component in the project workflow is in the tool mode and plugged into the agent
4. ask question about files in ingested dirs in chat and receive accurate answer
5. enjoy!

# How to build:

1. create knowledge outside of project by ingesting local dir with text files. Call it "short". Use ollama nomic-embed-text llm to create vector database from text files.
2. in your project workspace, create a custom component Local folder loader, that get files from selected dir
3. then parse those files in the chunks with Split text component
4. ingest these chunks into knowledge component, select Mode "Ingest" and Kowledge "short".
5. after that, switch your Knowledge component to tool mode for agent
6. create custom Local file search component, that can do exact and regex searches of files, turn on tool mode
7. change the code of Agent component to don't use Calculator, Current Date and Handle Parse Errors, as these are not necessary for the function and use more tokens. Use ollama local qwen3:1.7b llm. Restrict its behavior with system prompt

"You have two tools:

LOCAL FILE SEARCH
Use for:
- finding files
- filenames
- file extensions
- checking whether a file exists
- exact text/code searches

KNOWLEDGE
Use for:
- understanding document content
- summaries
- topics
- what a document is about

Rules:

1. For file existence or file type, use Local File Search first.

2. For exact text/code, use Local File Search.

3. For document meaning or topic, use Knowledge.

4. A question may require BOTH tools.

Example:
"Is there an HTML file, and what is it about?"

First call Local File Search to find HTML files.
If a file is found, copy its EXACT filename.
Then call Knowledge using that exact filename to retrieve its content.
Then answer from the tool results.

Never invent filenames.
Never claim a file does not exist without using Local File Search.
Never replace a filename returned by a tool with a shorter name."
