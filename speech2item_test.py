import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os
import shutil

class ItemExtractorNode():
    def __init__(self):
        model_name = "meta-llama/Llama-3.2-1B"
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        self.loop_count = 0
        
    def clear_cahce(self): 
        cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print("All Hugging Face model caches cleared.")
        else:
            print("No Hugging Face cache directory found.")
            
    def word_count(self, item):
        num_word = len(item.split())
        
        if num_word <= 3:
            return True
        else:
            return False
    
    def user_input_callback(self, msg):
        user_message = msg
        identified_item = self.identify_item(user_message)
        print(f"The item the user is looking for is: {identified_item}")
        
        if not self.word_count(identified_item) and self.loop_count < 3:
            self.loop_count += 1
            self.user_input_callback(msg)
            
        if self.loop_count >= 3:
            print("Model keeps making sentences clearing cache")
            self.clear_cahce()
        
        self.loop_count = 0
        
    def create_prompt(self, user_query):  
        return f"What is the name and description of the item trying to be found in the following sentence: '{user_query}' \n respond with (description of item) (item) only Answer: "

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                num_return_sequences=1
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def extract_item(self, response):
        print("*********")
        print(f"response \n {response}")
        
        #Get first answer
        first_answer = (response.split("\n")[1]).split("Answer: ")[-1]
        print("*********")
        print(f"first answer\n {first_answer}")
        
        if "2. " in first_answer:
            first_answer = first_answer.split("2. ")[0]
        if "2) " in first_answer:
            first_answer = first_answer.split("2) ")[0]
            
        print(f" trimed first answer\n {first_answer}")
        
        #Get item
        numbered_list = [f"{i}) " for i in range(1, 10)]
        numbered_list2 = [f"{i}. " for i in range(1, 10)]
        numbered_list3 = [f"{i}: " for i in range(1, 10)]
        char_list = ["\n", "the", "a ", "A "] 
        removals = numbered_list + numbered_list2 + numbered_list3 + char_list
        for removal in removals:
            first_answer = first_answer.replace(removal, "")
        
        item = first_answer
        
        print("*********")
        return item

    def identify_item(self, user_query):
        prompt = self.create_prompt(user_query)
        response = self.generate_response(prompt)
        item = self.extract_item(response)
        return item

def main(args=None):
    node = ItemExtractorNode()
    node.user_input_callback("i'm trying to find a green mug")

if __name__ == '__main__':
    main()
