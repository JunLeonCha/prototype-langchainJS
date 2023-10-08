import fs from "fs"
import EnvService from "./EnvService.js";

// Import OpenAI LLM
import { ChatOpenAI } from "langchain/chat_models/openai";
import { RetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib"
import { OpenAIEmbeddings } from "langchain/embeddings/openai"
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";


// Import document loaders for different file formats
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { JSONLoader } from "langchain/document_loaders/fs/json";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";

// Import Tiktoken for token counting
import { Tiktoken } from "@dqbd/tiktoken/lite";
import { load } from "@dqbd/tiktoken/load";
import registry from "@dqbd/tiktoken/registry.json" assert { type: "json" };
import models from "@dqbd/tiktoken/model_to_encoding.json" assert { type: "json" };

class OpenAIService {

    template = "You're an assistant, you're name is J.A.R.V.I.S, the meaning of your name is Just A Rather Very Intelligent System, you're here to respond clearly and simply as possible"
    humanTemplate = "{text}"

    //Fictive database? it's what i understand here
    VECTOR_STORE_PATH = "Documents.index"

    ChatModelType() {
        return new ChatOpenAI({
            openAIApiKey: EnvService.openAI_API_KEY(),
            modelName: "gpt-3.5-turbo-16k",
        });
    }

    async Loader() {
        const loader = new DirectoryLoader(
            "documents",
            {
                ".csv": (path) => new CSVLoader(path, "/csv"),
                ".json": (path) => new JSONLoader(path, "/json"),
                ".pdf": (path) => new PDFLoader(path, "/pdf"),
                ".txt": (path) => new TextLoader(path, "/text"),
            }
        );
        const docs = await loader.load();
        return docs;
    }

    async calculateCost() {
        const modelName = "text-embedding-ada-002";
        const modelKey = models[modelName];
        console.log(modelKey)
        const model = await load(registry[modelKey])
        const encoder = new Tiktoken(
            model.bpe_ranks,
            model.special_tokens,
            model.pat_str,
        );
        const tokens = encoder.encode(JSON.stringify(await this.Loader()));
        const tokenCount = tokens.length;
        const ratePerThousandTokens = 0.0004;
        const cost = (tokenCount / 1000) * ratePerThousandTokens;
        encoder.free();
        return cost
    }

    async normalizeDocuments(docs) {
        return await docs.map((doc) => {
            if (typeof doc.pageContent === "string") {
                return doc.pageContent;
            } else if (Array.isArray(doc.pageContent)) {
                return doc.pageContent.join("\n");
            }
        })
    }

    async run(question) {
        console.log("Calculating cost...");
        const cost = await this.calculateCost();
        console.log(`Cost calculated: ${cost} $`)

        if (cost <= 1) {
            // Initialize the OpenaAI Language Model
            this.ChatModelType()
            let vectoreStore;
            console.log("Checking for exiting vectore store...")
            if (fs.existsSync(this.VECTOR_STORE_PATH)) {
                //load the existing vectore store
                console.log("loading existing vectore store...");
                vectoreStore = await HNSWLib.load(
                    this.VECTOR_STORE_PATH,
                    new OpenAIEmbeddings()
                );
                console.log("Vectore store loaded")
            } else {
                //create a new vectore store
                console.log("Creating new vectore store");
                const textSpliter = new RecursiveCharacterTextSplitter({
                    chunkSize: 1000
                })
                const normalizeDocuments = await this.normalizeDocuments(await this.Loader())
                const splitDocs = await textSpliter.createDocuments(normalizeDocuments)

                // Generate the vectore store from the documents
                vectoreStore = await HNSWLib.fromDocuments(
                    splitDocs,
                    new OpenAIEmbeddings()
                )

                // Save the vectore store to the specified path
                await vectoreStore.save(this.VECTOR_STORE_PATH)
                console.log("Vector store created.")
            }

            //Create a retrieval chain using the language model and vectore store
            console.log("Quering chain...");
            const chain = RetrievalQAChain.fromLLM(this.ChatModelType(), vectoreStore.asRetriever());
            //Query the retrieval chain with the specified question
            const res = await chain.call({ query: question });
            console.log(res)
            return res
        } else {
            console.log("The cost of embedding exceeds 1$. Skipping embeddings")
        }
    }
}
export default new OpenAIService();