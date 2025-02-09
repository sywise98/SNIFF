# SNIFF - Social-Robot Navigator for Identifying and Finding Forgotten Items  
**A multimodal robotic platform integrating Vision-Language Models (VLMs), Large Language Models (LLMs), and autonomous navigation for assistive human-robot interaction.**

## Key Features  
- **Natural Conversation Interface**  
  - Real-time speech-to-text with OpenAI Whisper  
  - Contextual dialogue handling using Llama-3.2 LLM  
  - Text-to-speech conversion via PyTTSX3  

- **Advanced Perception System**  
  - Open-set object detection with YOLO-World v8  
  - 1080p stereo vision with depth perception  
  - 1.3m elevated camera perspective  

- **Autonomous Navigation**  
  - ROS2 NavStack with SLAM-based mapping  
  - Waypoint-based search pattern  
  - Custom safety protocols for human proximity  

## System Architecture  
| Component | Technology Stack | Key Specs |
|-----------|------------------|-----------|
| Hardware | Modified TurtleBot3 + Orange Pi 5 Max | 1.5GHz NPU, 32GB RAM, Offboard CUDA Compute for VLM |
| Vision | YOLO-World v8 + OpenCV | 30FPS @ 640×480 |
| Navigation | ROS2 NavStack | 0.2m/s avg speed |
| Conversation | Llama-3.2 + Whisper | <2s response latency |

## Performance Highlights  
**State Success Rates (20 trials):**  
```markdown
- Idle Detection: 65% [95% CI: 43.6-86.4%]
- User Approach: 100% 
- Conversation Handling: 100%
- Object Detection: 80% [62.0-98.0%] 
- Final Navigation: 55% [32.6-77.4%]
```

**Object-Specific Accuracy:**  
| Object | Success Rate | Relative Size |
|--------|--------------|---------------|
| Backpack | 96% | Large |
| Phone | 90% | Small |
| Bowl | 80% | Medium |
| Water Bottle | 56.7% | Medium |

## Key Findings  
1. **Robust Spatial Generalization**  
   - No significant performance difference across locations (χ²=0.104, p=0.991)  
   - Table 1-4 success rates: 86.7%, 85%, 68%, 80%  

2. **Error Propagation Analysis**  
   - Strong correlation between early-stage success and final outcome (r=0.81)  
   - Navigation errors accounted for 45% of total failures  

3. **Model Limitations**  
   - Water bottle detection challenges due to reflective surfaces  
   - 35% idle state failures from environmental distractions  

## Future Directions  
- Implement multi-user detection algorithms  
- Develop dynamic search patterns beyond fixed waypoints  
- Integrate tactile feedback for object verification  
- Optimize model quantization for edge deployment  

**Technologies Used:** ROS2, PyTorch, YOLO-World, Llama-3.2, OpenCV  
**Hardware:** TurtleBot3, Orange Pi 5 Max, ELP Stereo Cam, Anker PowerConf S330  
