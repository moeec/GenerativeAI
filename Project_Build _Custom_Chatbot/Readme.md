Project: Build Your Own Custom Chatbot

In this project I will change the default dataset to build my own custom chatbot. You will build the chatbot manually using just basic packages like openai and pandas, 
not using frameworks like LangChain, in order to gain a deeper understanding of how these systems work "under the hood".

My scenario is synthesizers, I am will be responsible for selecting a data source (https://en.wikipedia.org/wiki/Synthesizer)
I will strip down data and turn it into embeddings - DONE
Then use use an OpenAI Embedding model, "text-embedding-ada-002" to embedded my question - DONE
Calculated cosine distances between the question and the dataset I created from the Wikipeda - DONE
incorporating the data source into the custom chatbot code, and writing questions to demonstrate the performance of the custom prompt.

I built a custom chatbot using OpenAI, starting from scratch with a dataset I created from a Wikipedia page on synthesizers. 
Using tools like pandas and the OpenAI API, I:

- Extracted and structured the data
- Integrated it into a custom chatbot interface
- Demonstrated the chatbot’s performance through targeted questions

This project helped me better understand retrieval-augmented generation and how to adapt LLMs to specific domains.

Questions I asked were:

What components are used to alter sounds on a Synthesizer
A: Components such as filters, envelopes

What music genres have been influenced by the Synthesizer?
A:Electronic, hip hop, disco,

I now understand how to structure and format external data so it can be used effectively in a custom chatbot powered by OpenAI. This opens up exciting possibilities for building domain-specific assistants for everything from customer support to educational tools.

