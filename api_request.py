# -!- coding: utf-8 -!-
import openai


class Decoder:
    def __init__(self, api_key):
        self.api_key = api_key

    def decode(self, input, model, max_length):
        response = self.decoder_for_gpt3(model, input, max_length)
        return response

    def decoder_for_gpt3(self, model, input, max_length):
        openai.api_key = self.api_key
        # openai.api_base = "https://api.chatanywhere.com.cn/v1"

        if model == "text-ada-001":
            engine = "text-ada-001"
        elif model == "text-babbage-001":
            engine = "text-babbage-001"
        elif model == "text-curie-001":
            engine = "text-curie-001"
        elif model == "text-davinci-001":
            engine = "text-davinci-001"
        elif model == "text-davinci-002":
            engine = "text-davinci-002"
        elif model == "text-davinci-003":
            engine = "text-davinci-003"
        else:
            raise ValueError("model is not properly defined ...")

        response = openai.Completion.create(
            engine=engine,
            prompt=input,
            max_tokens=max_length,
            temperature=0,
            stop=None
        )

        return response["choices"][0]["text"]