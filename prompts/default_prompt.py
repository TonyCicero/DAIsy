from prompts.prompt_interface import IPrompt


class DefaultPrompt(IPrompt):

    def build(self, prompt):
        return f'''
        <|system|>
        Your name is DAIsy. You often talk in a Kawaii style.
        <|user|>
        {prompt}
        <|assistant|>
        '''

    def clean(self, prompt, response):
        return response.replace(self.build(prompt), "")