import argparse
import time
from threading import Thread
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from torchvision import transforms

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

from common.registry import registry


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all(input_ids[:, -len(stop):] == stop).item():
                return True

        return False


CONV_VISION_Vicuna0 = Conversation(    
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human: ", "Assistant: "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

CONV_VISION_LLama2 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("<s>[INST] ", " [/INST] "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

CONV_VISION_minigptv2 = Conversation(
    system="",
    roles=("<s>[INST] ", " [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

class Chat:
    def __init__(self, model, vis_processor, device='cuda:0', stopping_criteria=None, noise_level=0.25, alpha=0.001, monte_carlo_size=100, batch_size=48, smoothing=None):
        
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        if smoothing is not None:
            print(f"loading chat with params noise level={noise_level}. alpha={alpha}. monte_carlo_size={monte_carlo_size}. batch_size={batch_size}")
        self.smoothing = smoothing(self.model, noise_level) if smoothing else None
        self.inner_img_list = []
        self.inner_text = None
        self._abstain = False
        self._alpha=alpha
        self._monte_carlo_size = monte_carlo_size
        self._batch_size = batch_size

        if stopping_criteria is not None:
            self.stopping_criteria = stopping_criteria
        else:
            stop_words_ids = [torch.tensor([2]).to(self.device)]
            self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    
    def ask(self, text, conv):
        if self.smoothing is not None:
            print('asking with smoothing')
            self.inner_text = text                
            prediction = self.smooth_decoder()
            if prediction == self.smoothing.ABSTAIN:
                self._abstain = True
                return
        
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.            
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])            
        else:
            conv.append_message(conv.roles[0], text)

    def answer_prepare(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
                       repetition_penalty=1.05, length_penalty=1, temperature=1.0, max_length=2000):
        conv.append_message(conv.roles[1], None)        
        prompt = conv.get_prompt()         
        embs = self.model.get_context_emb(prompt, img_list)
        
        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]

        generation_kwargs = dict(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=float(temperature),
        )
        return generation_kwargs

    def answer(self, conv, img_list, **kargs):
        if self.smoothing is not None:
            if self._abstain:
                self._abstain = False
                print('abstain')
                return "abstain", ""
        
        generation_dict = self.answer_prepare(conv, img_list, **kargs)
        output_token = self.model_generate(**generation_dict)[0]
        output_text = self.model.llama_tokenizer.decode(output_token, skip_special_tokens=True)

        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()

        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def stream_answer(self, conv, img_list, **kargs):
        generation_kwargs = self.answer_prepare(conv, img_list, **kargs)
        streamer = TextIteratorStreamer(self.model.llama_tokenizer, skip_special_tokens=True)
        generation_kwargs['streamer'] = streamer
        thread = Thread(target=self.model_generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def model_generate(self, *args, **kwargs):
        # for 8 bit and 16 bit compatibility
        with self.model.maybe_autocast():            
            output = self.model.llama_model.generate(*args, **kwargs)                   
        return output

    def encode_img(self, img_list):
        print('uploading image')
        image = img_list[0]
        img_list.pop(0)
        if self.inner_img_list:
            self.inner_img_list.pop(0)
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
            print('image is a path')
        elif isinstance(image, Image.Image):            
            print('image is a PIL')
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
            if self.smoothing is not None: 
                self.inner_img_list.append(image)
        elif isinstance(image, torch.Tensor):
            print('image is a tensor')
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
            if self.smoothing is not None: 
                self.inner_img_list.append(image)

        image_emb, _ = self.model.encode_img(image)
        img_list.append(image_emb)

    def upload_img(self, image, conv, img_list):
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        img_list.append(image)        
        msg = "Received."

        return msg

    def forward_encoder(self, image_batch):
        if isinstance(image_batch, Image.Image):
            image_batch = self.vis_processor(image_batch).unsqueeze(0).to(self.device)
        elif isinstance(image_batch, torch.Tensor):
            if len(image_batch.shape) == 3:
                image_batch = image_batch.unsqueeze(0)
            image_batch = image_batch.to(self.device)
        else: 
            raise ValueError("Unsupported image type for forward_encoder")
                
        image_emb, _ = self.model.encode_img(image_batch)

        return image_emb       

    def get_img_list(self, image, img_list=[]):        
        img_list=[image]
        self.encode_img(img_list)
        return img_list
    
    def get_mixed_embs(self, args, img_list):
        prompt = args.prompt if hasattr(args, 'prompt') else "Describe this image: <ImageHere>."
        conv = Conversation(
            system="",
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        embs = self.model.get_context_emb(prompt, img_list)
        return embs
    
    def get_text(self, mixed_embs):
        generation_kwargs = dict(
            inputs_embeds=mixed_embs,
            max_new_tokens=20,
            stopping_criteria=self.stopping_criteria,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
        )
        with self.model.maybe_autocast():
            output_token = self.model.llama_model.generate(**generation_kwargs)[0]
        output_text = self.model.llama_tokenizer.decode(output_token, skip_special_tokens=True)
        return [output_text.strip()]
    
    def smooth_decoder(self):        
        message = f"[vqa] Based on the image, respond to this question in English with with a short answer: {self.inner_text}"        
        instruction = "<Img><ImageHere></Img> {} ".format(message)        
        data = {
            "image": self.inner_img_list[0],
            "question_id": 0,
            "instruction_input": [instruction],
            "answer": "",
            "image_id": 0
        }

           
        prediction = self.smoothing.predict(
            data, self._monte_carlo_size, self._alpha, batch_size=self._batch_size
        )

        return prediction
        
    