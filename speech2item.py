import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from llama_ros.langchain import LlamaROS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ItemExtractorNode(Node):
    def __init__(self):
        super().__init__('item_extractor_node')
        self.subscription = self.create_subscription(
            String,
            'user_input',
            self.user_input_callback,
            10
        )
        self.publisher = self.create_publisher(String, 'extracted_item', 10)
        
        # Initialize Llama 3 model
        self.llm = LlamaROS()
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["message"],
            template="Extract the item that needs to be found from this message: {message}\nItem:"
        )
        
        # Create chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def user_input_callback(self, msg):
        user_message = msg.data
        extracted_item = self.extract_item(user_message)
        
        output_msg = String()
        output_msg.data = extracted_item
        self.publisher.publish(output_msg)
        self.get_logger().info(f'Extracted item: {extracted_item}')

    def extract_item(self, message):
        return self.chain.invoke({"message": message})

def main(args=None):
    rclpy.init(args=args)
    node = ItemExtractorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
