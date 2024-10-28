import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class ItemExtractorNode():
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        
    
    def user_input_callback(self, msg):
        user_message = msg
        identified_item = self.identify_item(user_message)
        print(f"The item the user is looking for is: {identified_item}")
        
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
            first_answer.split("2. ")[0]
        if "2) " in first_answer:
            first_answer.split("2) ")[0]
        
        #Get item
        removals = ["\n", "the", "a ", "1) ", "1. "]
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
    node.user_input_callback("i'm looking for a green book")

if __name__ == '__main__':
    main()
