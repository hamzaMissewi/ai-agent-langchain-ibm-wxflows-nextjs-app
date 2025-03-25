import SYSTEM_MESSAGE from '@/constants/systemMessage';
import {AIMessage, BaseMessage, HumanMessage, SystemMessage, trimMessages} from '@langchain/core/messages';
import {ChatPromptTemplate, MessagesPlaceholder, PromptTemplate} from '@langchain/core/prompts';
import {END, MessagesAnnotation, START, StateGraph} from '@langchain/langgraph';
import {ToolNode} from '@langchain/langgraph/prebuilt';
import wxflows from '@wxflows/sdk/langchain';

import {FreeLLModelsEnum} from './types.js';
import {ChatOpenAI} from '@langchain/openai';
import {MemorySaver} from "@langchain/langgraph";

// import {HttpResponseOutputParser} from "langchain/output_parsers";
// import {ChatDeepSeek} from "@langchain/community/chat_models/deepseek";
// import {RunnableSequence} from "@langchain/core/runnables";
// import { ChatAnthropic } from "@langchain/anthropic";

// Trim the messages to manage conversation history
const trimmer = trimMessages({
    maxTokens: 10,
    strategy: "last",
    tokenCounter: (msgs) => msgs.length,
    includeSystem: true,
    allowPartial: false,
    startOn: "human"
});

const formatMessage = (message: BaseMessage) => {
    return `${message.name}: ${message.content}`;
};

// Connect to wxflows
const toolClient = new wxflows({
    endpoint: process.env.WXFLOWS_ENDPOINT || "",
    apikey: process.env.WXFLOWS_APIKEY
});

// Retrieve the tools
const tools = await toolClient.lcTools;
const toolNode = new ToolNode(tools);

const ANSWER_TEMPLATE_2 =
    `Role: You are a helpful assistant for NeuroMastery Bootcamp by Dr. Siddharth Warrier. 
Answer the question based only on the following context and chat history:
<context>
  {context}
</context>

<chat_history>
  {chat_history}
</chat_history>

Question: {question}
`;

const ANSWER_TEMPLATE = `Answer the user's questions based only on the following context. If the answer is not in the context, reply politely that you do not have that information available.:
==============================
Context: {context}
==============================
Current conversation: {chat_history}

user: {question}
assistant:`;


const PROMPT_FOOTER_CHAT = `
Current conversation:
{chat_history}

User: {input}
AI:`;


// Connect to the LLM provider with better tool instructions
const initialiseModel = (message: string) => {
    // const stream = await groq.chat.completions.create({
    //     // messages: enhancedMessages,
    //     model: FreeLLModelsEnum.llama3,
    //     stream: true,
    //     max_tokens: 1024,
    //     temperature: 0.7
    //   });

    // const todoGroq = await groq.chat.completions.create({
    //     messages: enhancedMessages,
    //     model: FreeLLModelsEnum.deepseek_llama,
    //     stream: true,
    //     max_tokens: 2024,
    //     tools,
    //     temperature: 0.5
    // });


    // const model = new ChatDeepSeek({
//   const model = new ChatGroq({
    const model = new ChatOpenAI({
        model: FreeLLModelsEnum.deepseek,
        apiKey: process.env.GROQ_API_KEY,
        temperature: 0.5,
        maxTokens: 1096,
        // streaming: true,
        // clientOptions: {
        //   defaultHeaders: {
        //     "anthropic-beta": "prompt-caching-2024-07-31"
        //   }
        // },
        callbacks: [
            {
                handleLLMStart: async () => {
                    // console.log("🤖 Starting LLM call");
                },
                handleLLMEnd: async (output) => {
                    console.log("🤖 End LLM call", output);
                    const usage = output.llmOutput?.usage;
                    if (usage) {
                        console.log("📊 Token Usage:", {
                            input_tokens: usage.input_tokens,
                            output_tokens: usage.output_tokens,
                            total_tokens: usage.input_tokens + usage.output_tokens,
                            cache_creation_input_tokens:
                                usage.cache_creation_input_tokens || 0,
                            cache_read_input_tokens: usage.cache_read_input_tokens || 0
                        });
                    }
                },
                handleLLMNewToken: async (token: string) => {
                    console.log("🔤 New token:", token);
                },
            }
        ]
    }).bindTools(tools);


    const response = await model.invoke(message);

    return response
    // return model;
};

// Define the function that determines whether to continue or not
function shouldContinue(state: typeof MessagesAnnotation.State) {
    const messages = state.messages;
    const lastMessage = messages[messages.length - 1] as AIMessage;

    // If the LLM makes a tool call, then we route to the "tools" node
    if (lastMessage.tool_calls?.length) {
        return "tools";
    }

    // If the last message is a tool message, route back to agent
    if (lastMessage.content && lastMessage._getType() === "tool") {
        return "agent";
    }

    // Otherwise, we stop (reply to the user)
    return END;
}

// Define a new graph
const createWorkflow = () => {
    const model = initialiseModel();

    return new StateGraph(MessagesAnnotation)
        .addNode("agent", async (state) => {
            // Create the system message content
            const systemContent = SYSTEM_MESSAGE + " " + PROMPT_FOOTER_CHAT;


            // Create the prompt template with system message and messages placeholder
            const promptTemplate = ChatPromptTemplate.fromMessages([
                new SystemMessage(systemContent, {
                    cache_control: {type: "ephemeral"}
                }),
                new MessagesPlaceholder("messages")
            ]);


            // Trim the messages to manage conversation history
            const trimmedMessages = await trimmer.invoke(state.messages);

            // Format the prompt with the current messages
            const prompt = await promptTemplate.invoke({messages: trimmedMessages});
            // Get response from the model
            const response = await model.invoke(prompt);
            return {messages: [response]};
        })
        .addNode("tools", toolNode)
        .addEdge(START, "agent")
        .addConditionalEdges("agent", shouldContinue)
        .addEdge("tools", "agent");
};

function addCachingHeaders(messages: BaseMessage[]): BaseMessage[] {
    if (!messages.length) return messages;

    // Create a copy of messages to avoid mutating the original
    const cachedMessages = [...messages];

    // Helper to add cache control
    const addCache = (message: BaseMessage) => {
        message.content = [
            {
                type: "text",
                text: message.content as string,
                cache_control: {type: "ephemeral"}
            }
        ];
    };

    // Cache the last message
    // console.log("🤑🤑🤑 Caching last message");
    addCache(cachedMessages.at(-1)!);

    // Find and cache the second-to-last human message
    let humanCount = 0;
    for (let i = cachedMessages.length - 1; i >= 0; i--) {
        if (cachedMessages[i] instanceof HumanMessage) {
            humanCount++;
            if (humanCount === 2) {
                // console.log("🤑🤑🤑 Caching second-to-last human message");
                addCache(cachedMessages[i]);
                break;
            }
        }
    }

    return cachedMessages;
}

export async function submitQuestion(messages: BaseMessage[], chatId: string) {
    // Add caching headers to messages
    try {
        const cachedMessages = addCachingHeaders(messages);
        console.log("🔒🔒🔒 Messages:", cachedMessages);

        // Create workflow with chatId and onToken callback
        const workflow = createWorkflow();

        // TODO TEST THIS
        // const model = initialiseModel();
        // const outputParser = new HttpResponseOutputParser();
        // const systemContent = SYSTEM_MESSAGE + " " + PROMPT_FOOTER_CHAT;
        // const systemPrompt = PromptTemplate.fromTemplate(systemContent);
        // const formattedPreviousMessages = cachedMessages.slice(0, -1).map(formatMessage);
        // const currentMessageContent = cachedMessages[cachedMessages.length - 1].content;

        // TODO
        // const chain = RunnableSequence.from([systemPrompt, model, outputParser]);
        // const chain = systemPrompt.pipe(model)//.pipe(outputParser);
        // const stream = await chain.stream({
        //     messages: cachedMessages,
        //     chat_history: formattedPreviousMessages.join("\n"),
        //     input: currentMessageContent,
        // }, {
        //     version: "v2",
        //     configurable: {thread_id: chatId},
        //     streamMode: "messages",
        //     runId: chatId,
        // })

        const checkpointer = new MemorySaver();
        const app = workflow.compile({checkpointer});

        const stream = await app.streamEvents(
            {messages: cachedMessages},
            {
                version: "v2",
                configurable: {thread_id: chatId},
                streamMode: "messages",
                runId: chatId,
            }
        );

        return stream;
        // return streamText.toDataStreamResponse(stream)
        // return StreamingTextResponse(stream);
    } catch (error) {
        throw new Error(`Error to submit question, Error ${JSON.stringify(error)}`);
    }
}
