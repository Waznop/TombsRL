import copy
import numpy as np
import enum
import random
import pickle
import os

class Faction(enum.Enum):

	GREY = enum.auto()
	LIGHT = enum.auto()
	DARK = enum.auto()

	@staticmethod
	def getOppFaction(faction):
		if faction == Faction.LIGHT:
			return Faction.DARK
		elif faction == Faction.DARK:
			return Faction.LIGHT
		else:
			raise Exception("Bad faction")

	@staticmethod
	def getAvailable():
		return [ Faction.LIGHT, Faction.DARK ]

class Role(enum.Enum):

	NONE = enum.auto()
	FIGHTER = enum.auto()
	CASTER = enum.auto()
	ROGUE = enum.auto()
	SUMMONER = enum.auto()

	@staticmethod
	def getAvailable():
		return [ Role.FIGHTER, Role.CASTER, Role.ROGUE, Role.SUMMONER ]

class Tile(enum.Enum):

	FREE = enum.auto()
	TOMB = enum.auto()

	KNIGHT = enum.auto()
	MAGICIAN = enum.auto()
	ASSASSIN = enum.auto()
	PRIEST = enum.auto()

	BERSERKER = enum.auto()
	SORCERER = enum.auto()
	BANDIT = enum.auto()
	NECROMANCER = enum.auto()

	@staticmethod
	def getFaction(tile):
		if tile == Tile.FREE or tile == Tile.TOMB:
			return Faction.GREY
		elif tile == Tile.KNIGHT or tile == Tile.MAGICIAN or tile == Tile.ASSASSIN or tile == Tile.PRIEST:
			return Faction.LIGHT
		elif tile == Tile.BERSERKER or tile == Tile.SORCERER or tile == Tile.BANDIT or tile == Tile.NECROMANCER:
			return Faction.DARK
		else:
			raise Exception("Bad tile")

	@staticmethod
	def getRole(tile):
		if tile == Tile.FREE or tile == Tile.TOMB:
			return Role.NONE
		elif tile == Tile.KNIGHT or tile == Tile.BERSERKER:
			return Role.FIGHTER
		elif tile == Tile.MAGICIAN or tile == Tile.SORCERER:
			return Role.CASTER
		elif tile == Tile.ASSASSIN or tile == Tile.BANDIT:
			return Role.ROGUE
		elif tile == Tile.PRIEST or tile == Tile.NECROMANCER:
			return Role.SUMMONER
		else:
			raise Exception("Bad role")

	@staticmethod
	def getTile(faction, role):
		if role == Role.FIGHTER:
			return Tile.KNIGHT if faction == Faction.LIGHT else Tile.BERSERKER
		elif role == Role.CASTER:
			return Tile.MAGICIAN if faction == Faction.LIGHT else Tile.SORCERER
		elif role == Role.ROGUE:
			return Tile.ASSASSIN if faction == Faction.LIGHT else Tile.BANDIT
		elif role == Role.SUMMONER:
			return Tile.PRIEST if faction == Faction.LIGHT else Tile.NECROMANCER
		else:
			raise Exception("Bad tile")

class TombsGame:

	HAND_SIZE = 2
	DECK_SIZE = 20
	BOARD_SIZE = (4, 4)
	INIT_TOMBS = 4
	CHECKPOINTS_DIR = "checkpoints"

	def __init__(self, seed=None):
		random.seed(seed)

		self._curFaction = Faction.LIGHT
		self._winner = Faction.GREY
		self._gameEnded = False
		self._board = np.full(self.BOARD_SIZE, Tile.FREE)
		self._decks = [], []
		self._hands = [], []
		self._scores = [0, 0]

		self._initBoard()
		self._initDecks()
		self._initHands()

		self._history = [(
			self._curFaction,
			self.getTurn(),
			np.copy(self._board),
			np.copy(self.getHand()),
			np.copy(self._scores)
		)]

	@property
	def board(self):
		return self._board

	@property
	def hands(self):
		return self._hands

	@property
	def decks(self):
		return self._decks

	@property
	def scores(self):
		return self._scores

	@property
	def history(self):
		return self._history

	@property
	def curFaction(self):
		return self._curFaction

	@property
	def winner(self):
		return self._winner

	@property
	def gameEnded(self):
		return self._gameEnded

	def _initBoard(self):
		row, col = self.BOARD_SIZE
		# tombs = random.sample(range(row * col), self.INIT_TOMBS)
		tombs = [0, 3, 12, 15]
		for t in tombs:
			self._board[t//col, t%col] = Tile.TOMB

	def _initDecks(self):
		light, dark = self._decks
		choices = Role.getAvailable()
		n = len(choices)
		for i in range(self.DECK_SIZE):
			idx = i % n
			light.append(choices[idx])
			dark.append(choices[idx])
		random.shuffle(light)
		random.shuffle(dark)

	def _initHands(self):
		for i in range(self.HAND_SIZE):
			for faction in Faction.getAvailable():
				self._draw(faction)

	def _isInBounds(self, coord):
		r, c = coord
		br, bc = self.BOARD_SIZE
		return r >= 0 and r < br and c >= 0 and c < bc

	def _getThreatenedCoords(self, coord, role, onAttack=False):
		r, c = coord
		coords = []
		if role == Role.SUMMONER and not onAttack:
			coords = []
		elif role == Role.ROGUE:
			coords = [coord]
		elif role == Role.FIGHTER or role == Role.SUMMONER:
			coords = [(r-1, c), (r, c+1), (r+1, c), (r, c-1)]
		elif role == Role.CASTER:
			coords = [(r-2, c), (r-1, c+1), (r, c+2), (r+1, c+1), (r+2, c), (r+1, c-1), (r, c-2), (r-1, c-1)]
		else:
			raise Exception("Bad role")
		return filter(self._isInBounds, coords)

	def _getHand(self, faction):
		if faction == Faction.LIGHT:
			return self._hands[0]
		elif faction == Faction.DARK:
			return self._hands[1]
		else:
			raise Exception("Bad faction")

	def _getDeck(self, faction):
		if faction == Faction.LIGHT:
			return self._decks[0]
		elif faction == Faction.DARK:
			return self._decks[1]
		else:
			raise Exception("Bad faction")

	def _getScore(self, faction):
		if faction == Faction.LIGHT:
			return self._scores[0]
		elif faction == Faction.DARK:
			return self._scores[1]
		else:
			raise Exception("Bad faction")

	def _increaseScore(self, faction):
		if faction == Faction.LIGHT:
			self._scores[0] += 1
		elif faction == Faction.DARK:
			self._scores[1] += 1
		else:
			raise Exception("Bad faction")

	def _getMoves(self, faction):
		opp = Faction.getOppFaction(faction)
		threatened = []
		enemyCoords = []
		tombCoords = []
		freeCoords = []

		br, bc = self.BOARD_SIZE
		for r in range(br):
			for c in range(bc):
				tile = self._board[r, c]
				if Tile.getFaction(tile) == opp:
					threatened += self._getThreatenedCoords((r, c), Tile.getRole(tile))
					enemyCoords.append((r, c))
				elif tile == Tile.TOMB:
					tombCoords.append((r, c))
				elif tile == Tile.FREE:
					freeCoords.append((r, c))
		threatened = set(threatened)
		enemyCoords = set(enemyCoords) - threatened;
		tombCoords = set(tombCoords) - threatened;
		freeCoords = set(freeCoords) - threatened;

		moves = []
		hand = self._getHand(faction)
		for idx, role in enumerate(hand):
			for coord in freeCoords:
				moves.append((idx, coord))
			if role == Role.ROGUE or role == Role.SUMMONER:
				for coord in tombCoords:
					moves.append((idx, coord))
			if role == Role.ROGUE:
				for coord in enemyCoords:
					moves.append((idx, coord))

		return moves

	def _draw(self, faction):
		hand = self._getHand(faction)
		deck = self._getDeck(faction)
		hand.append(deck.pop())

	def _makeMove(self, faction, idx, coord):
		moves = self._getMoves(faction)
		if (idx, coord) not in moves:
			return False

		r, c = coord
		opp = Faction.getOppFaction(faction)
		hand = self._getHand(faction)
		role = hand.pop(idx)

		oldTile = self._board[r, c]
		if oldTile == Tile.TOMB:
			self._increaseScore(faction)
		self._board[r, c] = Tile.getTile(faction, role)

		isThreat = not (role == Role.SUMMONER and oldTile != Tile.TOMB)
		threaten = self._getThreatenedCoords(coord, role, isThreat)
		for r, c in threaten:
			tile = self._board[r, c]
			if role == Role.SUMMONER:
				if Tile.getFaction(tile) != faction:
					if tile == Tile.TOMB:
						self._increaseScore(faction)
					self._board[r, c] = Tile.FREE
			elif Tile.getFaction(tile) == opp:
				self._board[r, c] = Tile.TOMB

		self._history.append((
			self._curFaction,
			self.getTurn(),
			np.copy(self._board),
			np.copy(self.getHand()),
			np.copy(self._scores)
		))
		return True

	def _prepTurn(self, faction):
		opp = Faction.getOppFaction(faction)
		deck = self._getDeck(faction)
		if not deck:
			score = self._getScore(faction)
			oppScore = self._getScore(opp)
			if score > oppScore:
				self._winner = faction
			elif score < oppScore:
				self._winner = opp
			self._gameEnded = True
			return False
		self._draw(faction)

		self._history.append((
			self._curFaction,
			self.getTurn(),
			np.copy(self._board),
			np.copy(self.getHand()),
			np.copy(self._scores)
		))

		moves = self._getMoves(faction)
		if not moves:
			self._winner = opp
			self._gameEnded = True
			return False
		return True

	### PUBLIC FUNCTIONS ###

	def prepTurn(self):
		return self._prepTurn(self._curFaction)

	def makeMove(self, idx, coord):
		success = self._makeMove(self._curFaction, idx, coord)
		if success:
			self._curFaction = Faction.getOppFaction(self._curFaction)
		return success

	def getMoves(self):
		return self._getMoves(self._curFaction)

	def getHand(self):
		return self._getHand(self._curFaction)

	def getTurn(self):
		return len(self._getDeck(self._curFaction))

	def saveHistory(self, name="replay_m.pkl"):
		if not os.path.exists(self.CHECKPOINTS_DIR):
			os.makedirs(self.CHECKPOINTS_DIR)
		path = os.path.join(self.CHECKPOINTS_DIR, name)
		with open(path, 'wb') as f:
			pickle.dump(self._history, f)

	### DEBUG FUNCTIONS ###

	def printDebug(self):
		print("Score: {}".format(self.scores))
		br, bc = self.BOARD_SIZE
		for r in range(br):
			for c in range(bc):
				print("{:2d}".format(self.board[r, c].value), end=" ")
			print("")
		print("--------------------")
		moves = self.getMoves()
		print("{} moves possible".format(len(moves)))
		hand = self.getHand()
		for role in hand:
			print(role, end=" ")
		print("\n--------------------")

	def play(self):
		while True:
			if not self.prepTurn():
				break
			self.printDebug();
			try:
				i, r, c = [int(x) for x in input().split()]
				while not self.makeMove(i, (r, c)):
					print("Illegal move")
					i, r, c = [int(x) for x in input().split()]
			except KeyboardInterrupt:
				break
		print("Winner: {}".format(self.winner))

if __name__ == "__main__":
	tombsGame = TombsGame(12)
	tombsGame.play()
