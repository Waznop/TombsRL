from tombs import TombsGame, Role, Tile, Faction
from enum import Enum
from train import TombsEnv
from model import ActorCritic
import tensorflow as tf
import numpy as np
import pygame
import sys
import pickle
import os
import argparse
import random

C_WHITE = (255, 255, 255)
C_LIGHT = (224, 224, 224)
C_DARK = (100, 100, 100)
C_RED = (255, 0, 0)
C_GREEN = (0, 255, 0)

class Agent(Enum):
	HUMAN = 0
	RANDOM = 1
	BOT = 2

	@classmethod
	def exists(cls, value):
		return any(value == item.value for item in cls)

class TombsWrapper:

	def __init__(self, path):
		self._history = []
		self._curTurn = 0
		self._tombsGame = None

		if path and os.path.exists(path):
			with open(path, 'rb') as f:
				self._history = pickle.load(f)
		else:
			self._tombsGame = TombsGame()

	@property
	def tombsGame(self):
		return self._tombsGame

	def isPlaying(self):
		return self._tombsGame != None

	def getBoard(self):
		if self.isPlaying():
			return self._tombsGame.board
		else:
			f, t, b, h, s = self._history[self._curTurn]
			return b

	def getBoardSize(self):
		if self.isPlaying():
			return self._tombsGame.BOARD_SIZE
		else:
			f, t, b, h, s = self._history[self._curTurn]
			return b.shape

	def getHandSize(self):
		if self.isPlaying():
			return self._tombsGame.HAND_SIZE + 1
		else:
			f, t, b, h, s = self._history[self._curTurn]
			return len(h) + 1

	def getHand(self):
		if self.isPlaying():
			return self._tombsGame.getHand()
		else:
			f, t, b, h, s = self._history[self._curTurn]
			return h

	def getScores(self):
		if self.isPlaying():
			return self._tombsGame.scores
		else:
			f, t, b, h, s = self._history[self._curTurn]
			return s

	def getCurFaction(self):
		if self.isPlaying():
			return self._tombsGame.curFaction
		else:
			f, t, b, h, s = self._history[self._curTurn]
			return f

	def getTurn(self):
		if self.isPlaying():
			return self._tombsGame.getTurn()
		else:
			f, t, b, h, s = self._history[self._curTurn]
			return t

	def getGameEnded(self):
		if self.isPlaying():
			return self._tombsGame.gameEnded
		else:
			return self._curTurn >= len(self._history)

	def getMoves(self):
		if self.isPlaying():
			return self._tombsGame.getMoves()
		return []

	def getWinner(self):
		if self.isPlaying():
			return self._tombsGame.winner
		if self.getTurn() == 0:
			s1, s2 = self.getScores()
			if s1 > s2:
				return Faction.LIGHT
			elif s1 < s2:
				return Faction.DARK
			else:
				return Faction.GREY
		else:
			return Faction.getOppFaction(self.getCurFaction())

	def prepTurn(self):
		if self.isPlaying():
			return self._tombsGame.prepTurn()

	def backtrack(self):
		assert(not self.isPlaying())
		if self._curTurn > 0:
			self._curTurn -= 1

	def makeMove(self, idx, tile):
		if self.isPlaying():
			return self._tombsGame.makeMove(idx, tile)
		else:
			self._curTurn += 1
			if self.getGameEnded():
				self._curTurn -= 1
				return False
			else:
				return True

	def saveHistory(self):
		if self.isPlaying():
			print("Saving history...")
			self._tombsGame.saveHistory()
			print("Done!")


class Display:

	S_SIZE = [480, 640]
	B_SIZE = 40
	M_SIZE = 10
	G_FPS = 30
	G_NAME = "Tombs"
	B_SELECT = 2
	B_FACTION = 10

	def __init__(self, p1, p2, path=None, save=False):
		pygame.init()
		pygame.display.set_caption(self.G_NAME)
		self._screen = pygame.display.set_mode(self.S_SIZE)
		self._clock = pygame.time.Clock()
		self._assets = {}
		self._initAssets()

		self._players = (p1, p2)
		self._save = save
		print("Starting a game of {} vs {}".format(p1, p2))
		self._tombs = TombsWrapper(path)
		self._boardSize = self._tombs.getBoardSize()
		self._handSize = self._tombs.getHandSize()

		self._env = None
		self._session = None
		self._model = None
		if p1 == Agent.BOT or p2 == Agent.BOT:
			self._env = TombsEnv(self._tombs.tombsGame)
			self._session = tf.Session()
			self._model = ActorCritic(self._env.obs_space(), self._env.act_space())
			saver = tf.train.Saver()
			latest = tf.train.latest_checkpoint(TombsGame.CHECKPOINTS_DIR)
			saver.restore(self._session, latest)

		self._illegalMove = False
		self._selectedTile = (-1, -1)
		self._selectedCard = -1
		self._tiles = np.full(self._boardSize, None)
		self._cards = []
		self._initObjects()

		self._tombs.prepTurn()
		self._run()

	def _run(self):
		running = True
		gameEnded = False
		while running:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					if self._save:
						self._tombs.saveHistory()
					if self._session:
						self._session.close()
					running = False
				if event.type == pygame.MOUSEBUTTONUP:
					if not self._tombs.getGameEnded():
						if self._tombs.isPlaying():
							agent = self._getCurPlayer()
							if agent == Agent.HUMAN:
								self._checkCollisions(pygame.mouse.get_pos())
							elif agent == Agent.RANDOM:
								self._makeRandomMove()
							elif agent == Agent.BOT:
								self._makeBotMove()
						elif event.button == 1: # LEFT
							self._tombs.makeMove(None, None)
						elif event.button == 3: # RIGHT
							self._tombs.backtrack()
					elif not gameEnded:
						print("Game ended! Winner is {}.".format(self._tombs.getWinner()))
						gameEnded = True

			self._screen.fill(C_WHITE)
			self._drawBoard()
			self._drawHand()
			self._drawUI()
			pygame.display.update()
			self._clock.tick(self.G_FPS)
		pygame.quit()

	def _getCurPlayer(self):
		faction = self._tombs.getCurFaction()
		p1, p2 = self._players
		return p1 if faction == Faction.LIGHT else p2

	### INIT ###

	def _initAssets(self):
		self._assets[Tile.TOMB] = pygame.image.load("assets/tomb.png")
		self._assets[Tile.KNIGHT] = pygame.image.load("assets/knight.png")
		self._assets[Tile.MAGICIAN] = pygame.image.load("assets/magician.png")
		self._assets[Tile.ASSASSIN] = pygame.image.load("assets/assassin.png")
		self._assets[Tile.PRIEST] = pygame.image.load("assets/priest.png")
		self._assets[Tile.BERSERKER] = pygame.image.load("assets/berserker.png")
		self._assets[Tile.SORCERER] = pygame.image.load("assets/sorcerer.png")
		self._assets[Tile.BANDIT] = pygame.image.load("assets/bandit.png")
		self._assets[Tile.NECROMANCER] = pygame.image.load("assets/necromancer.png")

	def _initObjects(self):
		nr, nc = self._boardSize
		w, h = self.S_SIZE
		tileSize = (w - 2 * self.B_SIZE - (nr - 1) * self.M_SIZE) / nr
		offset = tileSize + self.M_SIZE

		for r in range(nr):
			for c in range(nc):
				x_off = self.B_SIZE + c * offset
				y_off = self.B_SIZE + r * offset
				rect = pygame.Rect(x_off, y_off, tileSize, tileSize)
				self._tiles[r, c] = rect

		border = (w - self._handSize * tileSize - (self._handSize - 1) * self.M_SIZE) / 2
		for i in range(self._handSize):
			x_off = border + i * offset
			y_off = h - self.B_SIZE - tileSize
			rect = pygame.Rect(x_off, y_off, tileSize, tileSize)
			self._cards.append(rect)
		
		for k, v in self._assets.items():
			self._assets[k] = pygame.transform.scale(v, (rect.w, rect.h))

	### DRAW ###

	def _drawBoard(self):
		nr, nc = self._boardSize
		board = self._tombs.getBoard()

		for r in range(nr):
			for c in range(nc):
				rect = self._tiles[r, c]
				pygame.draw.rect(self._screen, C_LIGHT, rect)

				tile = board[r, c]
				asset = self._assets.get(tile)
				if asset:
					self._screen.blit(asset, rect)

		if self._selectedTile != (-1, -1):
			r, c = self._selectedTile
			rect = self._tiles[r, c]
			color = C_RED if self._illegalMove else C_GREEN
			pygame.draw.rect(self._screen, color, rect, self.B_SELECT)

	def _drawHand(self):
		hand = self._tombs.getHand()

		for i in range(self._handSize):
			rect = self._cards[i]
			pygame.draw.rect(self._screen, C_LIGHT, rect)

			if i < len(hand):
				role = hand[i]
				tile = Tile.getTile(self._tombs.getCurFaction(), role)
				asset = self._assets.get(tile)
				if asset:
					self._screen.blit(asset, rect)

		if self._selectedCard != -1:
			rect = self._cards[self._selectedCard]
			pygame.draw.rect(self._screen, C_GREEN, rect, self.B_SELECT)

	def _drawUI(self):
		light, dark = self._tombs.getScores()
		screenRect = self._screen.get_rect()
		font = pygame.font.Font(None, 40)
		text = font.render("BLUE - {} vs {} - RED".format(light, dark), 1, C_DARK)
		textpos = text.get_rect()
		textpos.centerx = screenRect.centerx
		textpos.centery = (screenRect.centery + self.S_SIZE[1]) / 2
		self._screen.blit(text, textpos)

		font = pygame.font.Font(None, 20)
		text = font.render("T{}".format(self._tombs.getTurn()), 1, C_DARK)
		textpos = text.get_rect()
		textpos.centerx = screenRect.centerx
		textpos.centery = (screenRect.centery + self.S_SIZE[1]) / 2 - 20
		self._screen.blit(text, textpos)

	### INPUT ###

	def _makeBotMove(self):
		faction = self._env.faction()
		obs = self._env.get_obs()
		legal = self._env.get_legal()

		policy = self._session.run(
			[self._model.policy_output],
			feed_dict={
				self._model.observations: [obs],
				self._model.legal_actions: [legal],
				self._model.is_training: False
			}
		)

		policy = policy[0][0]
		action_idx = np.argmax(policy)
		idx, r, c = np.unravel_index(action_idx, policy.shape)
		self._selectedCard = idx
		self._selectedTile = (r, c)
		self._takeTurn()

	def _makeRandomMove(self):
		moves = self._tombs.getMoves()
		idx, coord = random.choice(moves)
		self._selectedCard = idx
		self._selectedTile = coord
		self._takeTurn()

	def _checkCollisions(self, pos):
		nr, nc = self._boardSize

		for r in range(nr):
			for c in range(nc):
				rect = self._tiles[r, c]
				if rect.collidepoint(pos):
					self._selectedTile = (r, c)
					self._takeTurn()
					return

		for i in range(self._handSize):
			rect = self._cards[i]
			if rect.collidepoint(pos):
				self._selectedCard = i
				return

	def _takeTurn(self):
		if self._selectedCard == -1:
			self._illegalMove = True
			return

		if not self._tombs.makeMove(self._selectedCard, self._selectedTile):
			self._illegalMove = True
			return

		self._selectedCard = -1
		self._selectedTile = (-1, -1)
		self._tombs.prepTurn()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Play a game of Tombs!\
		0: Human\
		1: Random\
		2: Bot\
	')

	parser.add_argument('-r', help='watch replay')
	parser.add_argument('-p1', type=int, help='agent for player 1')
	parser.add_argument('-p2', type=int, help='agent for player 2')
	parser.add_argument('-s', action='store_true', help='save history')
	args = parser.parse_args()

	if args.r:
		path = os.path.join(TombsGame.CHECKPOINTS_DIR, args.r)
		Display(path=path)
		exit(0)

	p1 = Agent.HUMAN
	p2 = Agent.HUMAN
	if Agent.exists(args.p1):
		p1 = Agent(args.p1)
	if Agent.exists(args.p2):
		p2 = Agent(args.p2)
	Display(p1, p2, save=args.s)


