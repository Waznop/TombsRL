from model import ActorCritic
from tombs import TombsGame, Role, Tile, Faction

import numpy as np
import tensorflow as tf
import os
import random

class TombsEnv:

	DEF_FAC = Faction.LIGHT

	def __init__(self, tombs=None):
		if tombs:
			self._tombs = tombs
		else:
			self.reset()

	@staticmethod
	def obs_space():
		nr, nc = TombsGame.BOARD_SIZE
		nhand = TombsGame.HAND_SIZE + 1
		nscores = len(Faction.getAvailable())
		nturns = 1
		return (None, nr * nc + nhand + nscores + nturns)

	@staticmethod
	def act_space():
		nr, nc = TombsGame.BOARD_SIZE
		nhand = TombsGame.HAND_SIZE + 1
		return (None, nhand, nr, nc)

	def reset(self):
		self._tombs = TombsGame()
		self._tombs.prepTurn()
		return self.get_obs()

	def get_legal(self):
		moves = self._tombs.getMoves()
		act_sp = TombsEnv.act_space()
		mask = np.zeros(act_sp[1:])
		for i, tile in moves:
			r, c = tile
			mask[i, r, c] = True
		return mask

	def get_obs(self):
		nr, nc = self._tombs.BOARD_SIZE
		board = self._tombs.board
		faction = self._tombs.curFaction
		opp = Faction.getOppFaction(faction)

		obs = []
		for r in range(nr):
			for c in range(nc):
				tile = board[r, c]
				if faction != self.DEF_FAC:
					tFaction = Tile.getFaction(tile)
					tRole = Tile.getRole(tile)
					if tFaction == faction:
						tile = Tile.getTile(opp, tRole)
					elif tFaction == opp:
						tile = Tile.getTile(faction, tRole)
				obs.append(tile.value)

		hand = self._tombs.getHand()
		for role in hand:
			obs.append(role.value)

		turn = self._tombs.getTurn()
		factions = Faction.getAvailable()
		own_score = self._tombs.scores[factions.index(faction)]
		opp_score = self._tombs.scores[factions.index(opp)]
		obs += [turn, own_score, opp_score]

		return obs

	def step(self, action):
		i, r, c = action
		assert self._tombs.makeMove(i, (r, c))
		self._tombs.prepTurn()
		return self.get_obs()

	def winner(self):
		return self._tombs.winner

	def done(self):
		return self._tombs.gameEnded

	def faction(self):
		return self._tombs.curFaction

	def save(self, name):
		self._tombs.saveHistory(name)

class Trainer:

	BUFFER_SIZE = int(1e5)
	BATCH_SIZE = 1
	GAMMA = 0.99
	PLAYS_PER_STEP = 10
	CHECKPOINT = int(1e5)
	MAX_STEPS = int(1e9)
	SAVE_NAME = "model"

	def __init__(self):
		self._buffer = {
			"actions": [],
			"observations": [],
			"legal_actions": [],
			"advantages": [],
			"rewards": []
		}

	def buffer_size(self):
		return len(self._buffer["actions"])

	def remember(self, acts, obs, legal, adv, rewards):
		self._buffer["actions"] += list(acts)
		self._buffer["observations"] += list(obs)
		self._buffer["legal_actions"] += list(legal)
		self._buffer["advantages"] += list(adv)
		self._buffer["rewards"] += list(rewards)

		# assert same length
		assert(len(set(map(len, self._buffer.values()))))

		# chop down to max size
		size = self.buffer_size()
		if size > self.BUFFER_SIZE:
			to_chop = size - self.BUFFER_SIZE
			for k, v in self._buffer.items():
				self._buffer[k] = v[to_chop:]

	def recall(self):
		start = random.randint(0, self.buffer_size() - self.BATCH_SIZE)
		end = start + self.BATCH_SIZE
		acts = self._buffer["actions"][start:end]
		obs = self._buffer["observations"][start:end]
		legal = self._buffer["legal_actions"][start:end]
		adv = self._buffer["advantages"][start:end]
		rewards = self._buffer["rewards"][start:end]
		return acts, obs, legal, adv, rewards

	def play_once(self, session, model):
		env = TombsEnv()

		buf = {}
		for fac in Faction.getAvailable():
			buf[fac] = {
				"actions": [],
				"observations": [],
				"legal_actions": [],
				"values": []
			}

		while not env.done():
			faction = env.faction()
			obs = env.get_obs()
			legal = env.get_legal()

			policy, value = session.run(
				[model.policy_output, model.value_output],
				feed_dict={
					model.observations: [obs],
					model.legal_actions: [legal],
					model.is_training: False
				}
			)

			assert (self.BATCH_SIZE == 1)
			policy = policy[0]
			value = value[0]

			idx = np.random.choice(
				policy.size,
				p=policy.ravel()
			)
			action = np.unravel_index(idx, policy.shape)
			env.step(action)

			buf[faction]["actions"].append(action)
			buf[faction]["observations"].append(obs)
			buf[faction]["legal_actions"].append(legal)
			buf[faction]["values"].append(value)

		for fac in Faction.getAvailable():
			final_reward = 1 if fac == env.winner() else -1
			n_turns = len(buf[fac]["actions"])

			rewards = np.array([
				final_reward * self.GAMMA ** (n_turns - i) for i in range(n_turns)
			])
			advantages = rewards - np.array(buf[fac]["values"])

			self.remember(
				buf[fac]["actions"],
				buf[fac]["observations"],
				buf[fac]["legal_actions"],
				advantages,
				rewards
			)

		return env

	def train(self, resume):
		graph = tf.Graph()
		with graph.as_default():
			model = ActorCritic(TombsEnv.obs_space(), TombsEnv.act_space())
			saver = tf.train.Saver()

		with tf.Session(graph=graph) as session:
			summary_writer = tf.summary.FileWriter("logs", graph=session.graph)

			if resume:
				latest = tf.train.latest_checkpoint(TombsGame.CHECKPOINTS_DIR)
				saver.restore(session, latest)
				step = int(latest.split("-")[1])+1
			else:
				session.run(model.init_op)
				step = 0

			while step < self.MAX_STEPS:
				for _ in range(self.PLAYS_PER_STEP):
					last_env = self.play_once(session, model)

				for _ in range(self.buffer_size() // self.BATCH_SIZE):
					acts, obs, legal, adv, rewards = self.recall()

					loss, _ = session.run(
						[model.loss, model.optimize],
						feed_dict={
							model.actions: acts,
							model.observations: obs,
							model.legal_actions: legal,
							model.advantages: adv,
							model.rewards: rewards,
							model.is_training: True
						}
					)

					if step % self.CHECKPOINT == 0:
						print("Saving for checkpoint {}...".format(step))
						last_env.save("replay_{}.pkl".format(step))
						saver.save(session, os.path.join(TombsGame.CHECKPOINTS_DIR, self.SAVE_NAME),
							global_step=step)

					step += 1

if __name__ == "__main__":
	trainer = Trainer()
	trainer.train(resume=True)
