from model import ActorCritic
from train import TombsEnv
from tombs import TombsGame, Faction
import tensorflow as tf
import numpy as np
import random
import argparse

def evaluate(session, model, faction):
	tombs = TombsGame()
	env = TombsEnv(tombs)
	tombs.prepTurn()

	while not env.done():
		curFac = env.faction()

		if curFac == faction:
			obs = env.get_obs()
			legal = env.get_legal()

			policy = session.run(
				[model.policy_output],
				feed_dict={
					model.observations: [obs],
					model.legal_actions: [legal],
					model.is_training: False
				}
			)

			policy = policy[0][0]
			action_idx = np.argmax(policy)
			action = np.unravel_index(action_idx, policy.shape)
			env.step(action)

		else:
			moves = tombs.getMoves()
			idx, coord = random.choice(moves)
			assert(tombs.makeMove(idx, coord))
			tombs.prepTurn()

	return env.winner() == faction

def run(n=100):
	graph = tf.Graph()
	with graph.as_default():
		model = ActorCritic(TombsEnv.obs_space(), TombsEnv.act_space())
		saver = tf.train.Saver()

	with tf.Session(graph=graph) as session:
		latest = tf.train.latest_checkpoint(TombsGame.CHECKPOINTS_DIR)
		saver.restore(session, latest)

		wins = 0
		losses = 0

		botFaction = TombsEnv.DEF_FAC
		for i in range(n):
			if evaluate(session, model, botFaction):
				wins += 1
				print("Game {}: bot wins.".format(i))
			else:
				losses += 1
				print("Game {}: bot loses.".format(i))
			botFaction = Faction.getOppFaction(botFaction)

		print("Final bot score: {} - {}".format(wins, losses))

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Make bot play against random n times')
	parser.add_argument('-n', type=int)
	args = parser.parse_args()

	if args.n:
		run(n=args.n)
	else:
		run()
