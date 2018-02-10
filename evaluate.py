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

	return (env.winner() == faction, tombs.getTurn() == 0)

def run(n, f):

	if not n:
		n = 100
	if not f:
		f = 0

	graph = tf.Graph()
	with graph.as_default():
		model = ActorCritic(TombsEnv.obs_space(), TombsEnv.act_space())
		saver = tf.train.Saver()

	with tf.Session(graph=graph) as session:
		latest = tf.train.latest_checkpoint(TombsGame.CHECKPOINTS_DIR)
		saver.restore(session, latest)

		wins = [0, 0]
		losses = [0, 0]

		botFaction = Faction.LIGHT if f < 2 else Faction.DARK
		print("Bot faction: {}".format(botFaction))
		print("Alternate: {}".format(f == 0))
		for i in range(n):
			win, deckout = evaluate(session, model, botFaction)

			if win:
				if deckout:
					wins[0] += 1
				else:
					wins[1] += 1
			else:
				if deckout:
					losses[0] += 1
				else:
					losses[1] += 1

			if f == 0:
				botFaction = Faction.getOppFaction(botFaction)

		print("Final bot score: {} - {}".format(sum(wins), sum(losses)))
		print("Wins by deckout vs stuck: {} - {}".format(wins[0], wins[1]))
		print("Losses by deckout vs stuck: {} - {}".format(losses[0], losses[1]))

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Make bot play against random n times')
	parser.add_argument('-n', type=int)
	parser.add_argument('-f', type=int, choices=range(3), help='0: alternate, 1: light, 2: dark')
	args = parser.parse_args()
	run(n=args.n, f=args.f)
