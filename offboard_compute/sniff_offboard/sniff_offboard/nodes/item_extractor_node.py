import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import os
import shutil
import ollama

class ItemExtractorNode(Node):
    def __init__(self):
        super().__init__('item_extractor_node')
        self.subscription = self.create_subscription(
            String,
            '/speech_text',
            self.user_input_callback,
            10)
        
        self.clear_subscription = self.create_subscription(
            Bool,
            '/clear_llama_cache',
            self.llama_cache_callback,
            10)
            
        self.publisher = self.create_publisher(String, 'extracted_item', 10)
        self.loop_count = 0
      
    def clear_cahce(self): 
        cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print("All Hugging Face model caches cleared.")
        else:
            print("No Hugging Face cache directory found.")
             
    def llama_cache_callback(self, msg):
        #Clear caches
        self.clear_cahce()
        

    def user_input_callback(self, msg):
        user_message = msg.data
        identified_item = self.identify_item(user_message)
        print(f"The item the user is looking for is: {identified_item}")
        
        if not self.word_count(identified_item) and self.loop_count < 3:
            self.loop_count += 1
            self.user_input_callback(msg)
            
        if self.loop_count >= 3:
            self.get_logger().error("Model keeps making sentences clearing cache")
            self.clear_cahce()
        
        self.loop_count = 0
        item_msg = String()
        item_msg.data = identified_item
        self.publisher.publish(item_msg)
        
    def word_count(self, item):
        num_word = len(item.split())
        
        if num_word <= 3:
            return True
        else:
            return False

    def extract_item(self, response):
        # print("*********")
        # print(f"response \n {response}")
        
        #Get first answer
        first_answer = response
        if "Answer: " in first_answer:
            first_answer = (response.split("\n")[1]).split("Answer: ")[-1]
        # print("*********")
        # print(f"first answer\n {first_answer}")
        
        if "2. " in first_answer:
            first_answer = first_answer.split("2. ")[0]
        if "2) " in first_answer:
            first_answer = first_answer.split("2) ")[0]
            
        # print(f" trimed first answer\n {first_answer}")
        
        #Get item
        numbered_list = [f"{i}) " for i in range(1, 10)]
        numbered_list2 = [f"{i}. " for i in range(1, 10)]
        numbered_list3 = [f"{i}: " for i in range(1, 10)]
        char_list = ["\n", "the", "a ", "A "] 
        removals = numbered_list + numbered_list2 + numbered_list3 + char_list
        for removal in removals:
            first_answer = first_answer.replace(removal, "")
        
        item = first_answer
        
        # print("*********")
        return item


    def identify_item(self, user_query):
        #create prompt
        print("check")
        content = f"What is the name and description of the item trying to be found in the following sentence: '{user_query}' respond with only the item" 
        prompt=[
                    {
                        'role': 'user',
                        'content': content,
                    },
                 ]
        
        #generate response
        print(f'user prompt: {content}')
        response = ollama.chat(model='llama3.2', messages=prompt)
        response = response['message']['content']
        
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
