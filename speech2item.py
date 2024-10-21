import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

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
        
        self.model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


    def user_input_callback(self, msg):
        user_message = msg.data
        extracted_item = self.extract_item(user_message)
        
        output_msg = String()
        output_msg.data = extracted_item
        self.publisher.publish(output_msg)
        self.get_logger().info(f'Extracted item: {extracted_item}')

    def create_prompt(user_query):
        return f"Identify the item the user is asking to find in the following query: '{user_query}'"

    def generate_response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def extract_item(response):
        # This is a simple extraction method. You might need to adjust it based on the model's output format.
        return response.split(":")[-1].strip()

    def identify_item(user_query):
        prompt = create_prompt(user_query)
        response = generate_response(prompt)
        item = extract_item(response)
        return item

    # Example usage
    user_query = "Where can I find a red umbrella?"
    identified_item = identify_item(user_query)
    print(f"The item the user is looking for is: {identified_item}")


def main(args=None):
    rclpy.init(args=args)
    node = ItemExtractorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
