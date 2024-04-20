
class DefaultPrompt:

    def build(self, prompt):
        return f'''
        role: system
        content: You are a friendly chat bot who always responds in the the style of a computer.
        role: user
        content: {prompt}
        '''
