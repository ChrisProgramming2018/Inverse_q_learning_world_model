from replay_buffer import ReplayBuffer
import cv2

memory =  ReplayBuffer((64, 64, 3), (1,), 50000, "cuda")
memory.load_memory("pacman_expert_memory-9000")




state = memory.obses[0]
next_state = memory.next_obses[0]
print(state.shape)

cv2.imshow('HelloWorld', state)
cv2.waitKey(0)
cv2.imshow('HelloWorld', next_state)
cv2.waitKey(0)
