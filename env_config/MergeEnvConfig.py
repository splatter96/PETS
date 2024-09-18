import torch
import numpy as np
from highway_env import utils

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.set_printoptions(threshold=10_000)
COLLISION_REWARD = 200
HIGH_SPEED_REWARD = 1
MERGING_LANE_COST = 4
HEADWAY_COST = 4
REWARD_SPEED_RANGE = [10, 30]
LANE_CHANGE_COST = 1
OFFRAMP_REWARD = 100
HEADWAY_TIME = 1.2

MERGE_RAMP_START_X = 230
MERGE_RAMP_END_X= 310
MERGE_RAMP_START_Y = 7
MERGE_RAMP_LENGTH = 80

OFFRAMP_START_X = 310
OFFRAMP_END_X = 390
OFFRAMP_START_Y = 8

VEHICLE_LENGTH = 5.0
VEHICLE_WIDTH = 2.0

class MergeEnvConfig:

  @staticmethod
  def get_reward(obs, action):

   obs = obs.cpu()
   # action = action.cpu()

   #only for testing here!!
   # obs = np.array(obs).flatten()
   # obs = obs[np.newaxis, ...]
   # action = action[np.newaxis, ...]

   # For 5 vehicles:
   # each vehicle: presence, xpos, ypos, xspeed, yspeed
   vehicle_x_pos = obs[0, 1].double()
   vehicle_y_pos = obs[0, 2].double()
   vehicle_speed = obs[0, 3].double()

   vehicle_x_speed = obs[0, 3].double()
   vehicle_y_speed = obs[0, 4].double()

   vehicle_heading = np.arctan(vehicle_y_speed/vehicle_x_speed)

   # Check if vehicle has crashed
   vehicle_crashed = False
   headway_distance = 60
   for j in range(4):
       vx = obs[0, (j+1)*5+1].double() + vehicle_x_pos
       vy = obs[0, (j+1)*5+2].double() + vehicle_y_pos

       heading = np.arctan(vy/vx)

       # hacky headway calculation:
       # TODO incorporate heading of the ego vehicle

       # same lane, assume lanes are horizontal
       if abs(vehicle_y_pos - vy) < 0.5 and vehicle_x_pos < vx:
           hd = vx - vehicle_x_pos
           if hd < headway_distance:
               headway_distance = hd

       if vehicle_crashed or utils.norm(np.array([vehicle_x_pos, vehicle_y_pos]), np.array([vx, vy])) > VEHICLE_LENGTH**2:
           continue

       rect = utils.middle_to_vertices([vehicle_x_pos, vehicle_y_pos], VEHICLE_LENGTH, VEHICLE_WIDTH, vehicle_heading)
       other_rect = utils.middle_to_vertices([vx, vy], VEHICLE_LENGTH, VEHICLE_WIDTH, heading)

       if utils.separating_axis_theorem(rect, other_rect):
           vehicle_crashed = True
           # break

   scaled_speed = utils.lmap(vehicle_speed, REWARD_SPEED_RANGE, [0, 1])

   # compute cost for staying on the merging lane
   if vehicle_x_pos > MERGE_RAMP_START_X and vehicle_x_pos < MERGE_RAMP_END_X and vehicle_y_pos > MERGE_RAMP_START_Y:
       Merging_lane_cost = - np.exp(-(vehicle_x_pos - MERGE_RAMP_END_X) ** 2 / (10 * MERGE_RAMP_LENGTH))
   else:
       Merging_lane_cost = 0

   # give penalty if the agent drives on the offramp
   if vehicle_x_pos > OFFRAMP_START_X and vehicle_x_pos < OFFRAMP_END_X and vehicle_y_pos > OFFRAMP_START_Y:
       offramp_cost = -OFFRAMP_REWARD
   else:
       offramp_cost = 0

   # compute headway cost
   Headway_cost = np.log(headway_distance / (HEADWAY_TIME * vehicle_speed)) if vehicle_speed > 0 else 0

   # compute overall reward
   reward = COLLISION_REWARD * (-1 * vehicle_crashed) \
             + (HIGH_SPEED_REWARD * np.clip(scaled_speed, 0, 1)) \
             + MERGING_LANE_COST * Merging_lane_cost \
             + HEADWAY_COST * (Headway_cost if Headway_cost < 0 else 0) \
             + offramp_cost

   combined_reward = np.full(obs.shape[0], reward)

   #parallel evaluation of Lane_change_cost
   action_discrete = torch.argmax(action, dim=1)
   Lane_change_cost = torch.where((action_discrete == 0) | (action_discrete == 2), -LANE_CHANGE_COST, 0.0)

   # combine parallel and sequentiall evaluated rewards
   combined_reward = torch.from_numpy(combined_reward)
   combined_reward += Lane_change_cost

   # return combined_reward
   return combined_reward.float().to(TORCH_DEVICE)

