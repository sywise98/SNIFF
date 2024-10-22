import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
#source .env/bin/activate

class ItemExtractorNode(Node):
    def __init__(self):
        super().__init__('item_extractor_node')
        self.subscription = self.create_subscription(
            String,
            '/speech_text',
            self.user_input_callback,
            10
        )
        self.publisher = self.create_publisher(String, 'extracted_item', 10)
        
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")


    def user_input_callback(self, msg):
        user_message = msg.data
        identified_item = self.identify_item(user_message)
        print(f"The item the user is looking for is: {identified_item}")
        
        item_msg = String()
        item_msg.data = identified_item
        self.publisher.publish(item_msg)


    def create_prompt(user_query):
        return f"Identify the item the user is asking to find in the following query: '{user_query}'"

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def extract_item(self, response):
        # This is a simple extraction method. You might need to adjust it based on the model's output format.
        print(f"response: {response}")
        return response.split(":")[-1].strip()

    def identify_item(self, user_query):
        prompt = self.create_prompt(user_query)
        response = self.generate_response(prompt)
        item = self.extract_item(response)
        return item

    

def main(args=None):
    rclpy.init(args=args)
    node = ItemExtractorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
