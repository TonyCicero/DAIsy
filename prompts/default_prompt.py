from prompts.prompt_interface import IPrompt


class DefaultPrompt(IPrompt):

    def build(self, prompt):
        return f'''
        <|system|>
        You are a friendly girl named DAIsy.
        <|user|>
        {prompt}
        <|assistant|>
        '''

    def clean(self, prompt, response):
        return response.replace(self.build(prompt), "")