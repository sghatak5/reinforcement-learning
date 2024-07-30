import gymnasium as gym
from gymnasium import spaces
import random
import pygame

class HanoiTower(gym.Env):
    """
    Tower of Hanoi environment.

    Args:
        numDisks (int): Number of disks in the Tower of Hanoi. Default is 3.

    Attributes:
        numDisks (int): Number of disks in the Tower of Hanoi.
        agentState (tuple): Current state of the Tower of Hanoi represented as a tuple.
        goalState (tuple): Goal state of the Tower of Hanoi represented as a tuple.
        actionSpace (gym.Space): Action space for the Tower of Hanoi environment.
        observationSpace (gym.Space): Observation space for the Tower of Hanoi environment.
        done (bool): Indicates whether the environment has reached a terminal state.
        actionLookup (dict): Dictionary mapping action indices to human-readable descriptions.
    
    Methods:
        reset(): Resets the environment to its initial state.
        step(action): Executes a step in the environment given an action.
        disksOnPeg(peg): Returns a list of disks on a given peg.
        moveAllowed(action): Checks if a given move is allowed.
        render(): Renders the current state of the Tower of Hanoi.
    """
    def __init__(self, numDisks = 3):
        """
        Initialize the Tower of Hanoi environment.

        Args:
            numDisks (int): Number of disks in the Tower of Hanoi. Default is 3.
        """
        super().__init__()
        self.numDisks = numDisks
        self.agentState = numDisks * (0, )
        self.goalState = numDisks * (2, ) 
        self.reward = 0
        self.actionSpace = spaces.Discrete(6)
        self.observationSpace = spaces.Tuple(self.numDisks*(spaces.Discrete(3) ,))
        self.done = None
        self.actionLookup = {0 : "(0,1) - top disk of pole 0 to top disk of pole 1 ",
                              1 : "(0,2) - top disk of pole 0 to top disk of pole 2 ",
                              2 : "(1,0) - top disk of pole 1 to top disk of pole 0 ",
                              3 : "(1,2) - top disk of pole 1 to top disk of pole 2 ",
                              4 : "(2,0) - top disk of pole 2 to top disk of pole 0 ",
                              5 : "(2,1) - top disk of pole 2 to top disk of pole 1 ",}

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            tuple: Initial state of the Tower of Hanoi.
        """
        self.agentState = self.numDisks * (0, ) 
        self.done = False
        
        return (self.agentState)
    
    def step(self, action):
        """
            Execute a step in the environment given an action.

            If the action is allowed (i.e., the move is valid according to the rules of the Tower of Hanoi),
            the agent's state is updated to reflect the move. The reward is incremented by 0.001 and rounded
            to three decimal places. If the updated state matches the goal state, the episode terminates
            with an additional reward of 1 (total reward incremented by 1) and the environment's `done` flag 
            is set to True, indicating that the goal has been reached. 

            If the action is not allowed (i.e., the move violates the rules of the Tower of Hanoi),
            the reward is decremented by 0.01 and rounded to three decimal places, and the `info` variable 
            is set to "Invalid action". No state update is performed in this case.

            Args:
                action (int): Index of the action to be executed.

            Returns:
                tuple: A tuple containing the next state (tuple), the cumulative reward (float), 
                    a boolean indicating if the episode has terminated, and an info string.
            """
        if self.moveAllowed(action):
            diskToMove = min(self.disksOnPeg(action[0]))
            movedState = list(self.agentState)
            movedState[diskToMove] = action[1]
            self.agentState = tuple(movedState)
            #self.reward = round(self.reward - 0.001, 3)
            self.reward = -0.001
            info = "Move was successfull but did not reach goal"
        else:
            #self.reward = round(self.reward - 0.1, 3)
            self.reward = -0.1
            info = "Invalid action"
        
        if self.agentState == self.goalState:
            #self.reward = round(self.reward + 1, 3)
            self.reward = +1
            self.done = True
            info = "Reached the goal"

        return self.agentState, self.reward, self.done, info

            

    def disksOnPeg(self, peg):
        """
        Get a list of disks on a given peg.

        Args:
            peg (int): Index of the peg.

        Returns:
            list: List of disk indices on the specified peg.
        """
        listOfDisk = []
        for disk in range(self.numDisks):
            if self.agentState[disk] == peg:
                listOfDisk.append(disk)
        return listOfDisk        
    
    def moveAllowed(self, action):
        """
        Check if a given move is allowed.

        Args:
            action (tuple): Action representing the move.

        Returns:
            bool: True if the move is allowed, False otherwise.
        """
        disks_from = self.disksOnPeg(action[0])
        disks_to = self.disksOnPeg(action[1])

        if disks_from:
            if disks_to:
                return min(disks_to) > min(disks_from)
            else:
                return True
        return False

    def render(self):
        """
        Render the current state of the Tower of Hanoi environment using Pygame.

        This method initializes a Pygame window and renders the current state
        of the Tower of Hanoi environment. It displays the pegs and disks
        using rectangles of different colors and sizes to represent the disks.
        Peg1 = Blue
        Peg2 = Red
        Peg3 = Green
        Whenever a disk is moved to a different peg, disk will be displayed in the respective peg color. 

        Note:
            This method should be called after making a successful move in the
            Tower of Hanoi environment:

        Returns:
            None
        """
        pygame.init()
        pygame.display.set_caption("Hanoi Tower")

        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 255, 255)

        size = (800, 400)
        screen = pygame.display.set_mode(size)
        clock = pygame.time.Clock()
        hanoi_surface = pygame.image.load('.\Project\HanoiTower_v5.png') # type: ignore

        disk_height = 25
        disk_width = 200    

        screen.blit(hanoi_surface, (0, 0))

        def initializeDisks(diskNumber, disk_height, disk_width):
            disks = []
            width = disk_width
            height = disk_height  

            for _ in range(diskNumber):
                disks.append([width, height])
                width -= 25

            return disks
            
        disk_info = initializeDisks(self.numDisks, disk_height, disk_width)
        peg_0 = 0
        peg_1 = 0
        peg_2 = 0

        for disk, peg in enumerate(self.agentState[::-1]):  # agentState reversed for proper rendering
            if peg == 0:
                pygame.draw.rect(screen, BLUE, (80 + (12.5 * disk), 329 - (25 * peg_0), disk_info[disk][0], disk_info[disk][1]))
                peg_0 += 1
            elif peg == 1:
                pygame.draw.rect(screen, RED, (300 + (12.5 * disk), 329 - (25 * peg_1), disk_info[disk][0], disk_info[disk][1]))
                peg_1 += 1
            else:
                pygame.draw.rect(screen, GREEN, (520 + (12.5 * disk), 329 - (25 * peg_2), disk_info[disk][0], disk_info[disk][1]))
                peg_2 += 1

        pygame.display.update()
        clock.tick(60)
        pygame.time.wait(1)


    def close(self):
        pygame.quit()

#Function that creates and instance of the environment
def create_env(numDisks):
    env = HanoiTower(numDisks=numDisks)

    return env