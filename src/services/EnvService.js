import dotenv from "dotenv"

class Env {
    constructor() {
        dotenv.config()
    }

    PORT() {
        return process.env.PORT || 8080
    }

    openAI_API_KEY() {
        return process.env.OPENAI_API_KEY
    }
}

export default new Env()