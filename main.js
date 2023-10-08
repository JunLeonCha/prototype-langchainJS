import express from "express"
import Env from "./src/services/EnvService.js"
import OpenAIService from "./src/services/OpenAIService.js"


const app = express()
const PORT = Env.PORT()

app.get('/chat', async (req, res) => {
    const Input = "Qui est Sakura Kimura et qu'est-ce qu'elle aime faire de son temps?"
    console.log("GET: /chat")
    const ChatEmbedding = await OpenAIService.run(Input)
    return res.send(ChatEmbedding)
})

app.listen(PORT, () => {
    console.log(`listening on http://localhost:${PORT}`)
})