# hybrid_trainer.py
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class HybridPPOTrainer:
    """
    A trainer that combines Reinforcement Learning (PPO) with Supervised Learning
    from expert demonstrations to accelerate training and improve performance.
    """
    
    def __init__(self, agent, optimizer, demonstrations, device, 
                 sl_weight=0.5, sl_decay=0.999, norm_adv=True, clip_coef=0.2,
                 clip_vloss=True, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5):
        """
        Initialize the hybrid PPO trainer.
        
        Args:
            agent: The agent to train
            optimizer: The optimizer to use
            demonstrations: List of expert demonstrations
            device: The device to use for tensor operations
            sl_weight: Initial weight for supervised learning loss
            sl_decay: Decay rate for supervised learning weight
            norm_adv: Whether to normalize advantages
            clip_coef: PPO clipping coefficient
            clip_vloss: Whether to clip value loss
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.agent = agent
        self.optimizer = optimizer
        self.device = device
        
        # PPO hyperparameters
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        # Supervised learning parameters
        self.sl_weight = sl_weight
        self.initial_sl_weight = sl_weight
        self.sl_decay = sl_decay
        
        # Process demonstrations
        self._process_demonstrations(demonstrations)
        
    def _process_demonstrations(self, demonstrations):
        """Process and prepare expert demonstrations for training"""
        if not demonstrations:
            print("Warning: No demonstrations provided for hybrid learning!")
            self.has_demonstrations = False
            return
            
        # Flatten and concatenate all demonstration data
        demo_states = []
        demo_actions = []
        
        for demo in demonstrations:
            demo_states.append(demo['states'])
            demo_actions.append(demo['actions'])
            
        self.demo_states = np.concatenate(demo_states)
        self.demo_actions = np.concatenate(demo_actions)
        
        # Convert to torch tensors
        self.demo_states = torch.FloatTensor(self.demo_states).to(self.device)
        self.demo_actions = torch.LongTensor(self.demo_actions).to(self.device)
        
        self.has_demonstrations = True
        print(f"Loaded {len(self.demo_states)} demonstration steps for hybrid learning")
    
    def sample_demonstrations(self, batch_size):
        """Sample a batch of demonstrations"""
        if not self.has_demonstrations:
            return None, None
            
        indices = np.random.randint(0, len(self.demo_states), batch_size)
        return self.demo_states[indices], self.demo_actions[indices]
    
    def compute_supervised_loss(self, states, expert_actions):
        """Compute supervised learning loss between predicted actions and expert actions"""
        logits = self.agent.actor(states)
        return nn.CrossEntropyLoss()(logits, expert_actions)
    
    def train_minibatch(self, b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values, mb_inds):
        """
        Train on a single minibatch with hybrid RL + SL loss.
        
        Args:
            b_obs: Batch of observations
            b_actions: Batch of actions
            b_logprobs: Batch of log probabilities
            b_advantages: Batch of advantages
            b_returns: Batch of returns
            b_values: Batch of values
            mb_inds: Indices of the minibatch
            
        Returns:
            Dictionary of loss metrics
        """
        # Standard PPO forward pass
        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
        logratio = newlogprob - b_logprobs[mb_inds]
        ratio = logratio.exp()

        # Calculate approx_kl and clipfracs
        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()

        # Normalize advantages
        mb_advantages = b_advantages[mb_inds]
        if self.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if self.clip_vloss:
            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
            v_clipped = b_values[mb_inds] + torch.clamp(
                newvalue - b_values[mb_inds],
                -self.clip_coef,
                self.clip_coef,
            )
            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

        # Entropy loss
        entropy_loss = entropy.mean()
        
        # Initialize total loss with standard PPO components
        loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
        
        # Add supervised learning component if demonstrations are available
        sl_loss = torch.tensor(0.0).to(self.device)
        if self.has_demonstrations and self.sl_weight > 0.001:  # Only if weight is significant
            demo_states, demo_actions = self.sample_demonstrations(len(mb_inds))
            sl_loss = self.compute_supervised_loss(demo_states, demo_actions)
            loss = loss + self.sl_weight * sl_loss

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Decay supervised learning weight
        self.sl_weight *= self.sl_decay
        
        return {
            "value_loss": v_loss.item(),
            "policy_loss": pg_loss.item(),
            "entropy": entropy_loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": clipfrac.item(),
            "supervised_loss": sl_loss.item(),
            "sl_weight": self.sl_weight
        }
        
    def get_current_sl_weight(self):
        """Get the current supervised learning weight"""
        return self.sl_weight
