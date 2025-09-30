# Pulpito_Luchador
'''
The objective of this project is to use ML techniques to identify joints on the human body, and then analyze them for positional advantages in the context of Brazilian Jiu Jitsu. 
'''

## Tools
'''
- Python
- OpenCV
- MediaPipe
'''

### Foundation
1) Data Collection
- Collect data using phone video camera
- Start with simple movements (shrimp, bridge, technical standup)
2) Pose Estimation Setup
- Use MediaPipe to extract keypoitns (hips, knewws, shoulders, etc.)
- Store joint angle/time in CSV or JSON
3) Form Analysis
- Calculate angles & distances (e.g. hip height from ground, knee flare during shrimp)
- Compare your reps to reference reps