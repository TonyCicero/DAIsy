
class DefaultPrompt:

    def build(self, prompt):
        return f'''
        <|system|>
        You are a helpful assistant.
        <|user|>
        {prompt}
        <|assistant|>
        '''

    def clean(self, prompt, response):
        return response.replace(self.build(prompt), "")