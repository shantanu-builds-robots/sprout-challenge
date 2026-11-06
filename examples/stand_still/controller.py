"""The smallest valid submission: hold the starting pose. It never falls and never finishes."""
from sprout_challenge import Controller


class StandStill(Controller):
    def reset(self, info):
        self.pose = info["default_pose"]

    def act(self, obs):
        return self.pose
