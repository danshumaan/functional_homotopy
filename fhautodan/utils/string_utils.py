import torch
from fastchat import model
from fastchat.conversation import get_conv_template
from icecream import ic

def load_conversation_template(template_name):
    if template_name!='llama-2' and template_name!='llama2':
        return get_conv_template(template_name)
    if template_name == 'llama2':
        template_name = 'llama-2'
    conv_template = model.get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
        conv_template.system = "[INST] <<SYS>>\n\n<</SYS>>\n\n"
    return conv_template


def compute_slice(tokenizer, tokens, prompt, string, template, part):
    if template=='llama-3' and part=='output':
        string_tokens = tokenizer(prompt[prompt.find(string)-1]+string, return_tensors="pt")['input_ids'][0][2:]
    elif template=='llama-3' and part=='perturbation':
        string_tokens = tokenizer(prompt[prompt.find(string)-1]+string, return_tensors="pt")['input_ids'][0][1:]
    elif template=='zephyr' and part=='output':
        string_tokens = tokenizer(prompt[prompt.find(string)-1]+string, return_tensors="pt")['input_ids'][0][3:]
    elif template=='gemma-2-9b-it' and part=='perturbation':
        string_tokens = tokenizer(prompt[prompt.find(string)-1]+string, return_tensors="pt")['input_ids'][0][1:]
    else:
        string_tokens = tokenizer(string, return_tensors="pt")['input_ids'][0][1:]
    # string_tokens = string_tokens # .to(device)
    slices = []

    for i, token in enumerate(tokens.flip(0)):
        i = len(tokens) - i - 1
        if token == string_tokens[0]:
            slice_start = i
            slice_end = i + 1
            # breakpoint()
            while slice_end <= len(tokens) and slice_end - slice_start <= len(string_tokens) and (
                    tokens[slice_start:slice_end] == string_tokens[:slice_end - slice_start]).all().item():
                slice_end += 1
            slice_end -= 1
            # breakpoint()
            if slice_end - slice_start == len(string_tokens) and (
                    tokens[slice_start:slice_end] == string_tokens).all().item():
                slices.append(slice(slice_start, slice_end))
    if len(slices) > 0:
        return slices
    else:
        raise ValueError("String not found in tokens.")

def format_prompt(adv_input, output, sys_prompt, template):
    conv = get_conv_template(template)
    if sys_prompt is not None:
        conv.set_system_message(sys_prompt)
    # adv_input has the task embedded within the adversarial msg.
    conv.append_message(conv.roles[0], "{adv_input}")
    conv.append_message(conv.roles[1], "{output}")
    text = conv.get_prompt()
    text = text.format(adv_input=adv_input, output=output)
    if template=='vicuna_v1.1':
        text = text.replace("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. ", "")
    return text

class autodan_SuffixManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):

        self.tokenizer = tokenizer
        self.template = conv_template
        self.conv_template = load_conversation_template(conv_template)
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string

    def get_prompt(self, adv_string=None):
        if adv_string is not None:
            self.adv_string = adv_string.replace('[REPLACE]', self.instruction.lower())
        
        # input = self.instruction
        self.output = self.target
        self.input = self.adv_string
        template = self.template
        """
        The input is task embedded within the adv_string!
        self.goal==task
        self.target==goal!
        """
        if template=='gemma-2-9b-it' or template=='mistral' or template=='llama-3':
            chat = [
                { "role": "user", "content": "{adv_string}" },
                { "role": "assistant", "content": "{output}" },
            ]
            self.prompt = self.tokenizer.apply_chat_template(chat, tokenize=False,) # add_generation_prompt=True)
            self.prompt = self.prompt.format(adv_string=self.input, output=self.output)
            self.tokens = self.tokenizer(self.prompt, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
            
        else:
            self.prompt = format_prompt(adv_input=self.input,
                                    output=self.output,
                                    sys_prompt=None,
                                    template=template)
            
            self.tokens = self.tokenizer(self.prompt, return_tensors="pt")['input_ids'][0]# .to(device)
        self.seq_len = len(self.tokens)
        
        # Input should contain the system message as well.
        self.input = self.prompt[:self.prompt.find(self.input)+len(self.input)]
        
        # Identify slices for question, perturbation, and response.
        self.input_slice = compute_slice(self.tokenizer, 
                                         self.tokens, 
                                         self.prompt, 
                                         self.input, 
                                         template, part='input')[0]
        
        self._goal_slice = self.input_slice
        self._control_slice = self._goal_slice
        
        self.output_slice = compute_slice(self.tokenizer, 
                                          self.tokens,  
                                          self.prompt, 
                                          self.output, 
                                          template, part='output')[0]
        
        self._target_slice = self.output_slice
        
        self._loss_slice = slice(self.output_slice.start - 1, self.output_slice.stop - 1)
        
        return self.prompt

    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        input_ids = self.tokens[:self._target_slice.stop]
        
        return input_ids
