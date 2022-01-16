import logging
from typing import List
import chess.pgn

def get_game_filter(game_filter_name: str):
	if game_filter_name == "filter_out_bullet_games":
		return filter_out_bullet_games
	else:
		raise ValueError(f"Unknown game_filter_name: {game_filter_name}")

def filter_out_bullet_games(header: chess.pgn.Headers, debug: bool = False):
	return filter_by_elo(header, [700, 3000], [700, 3000], debug) and \
		filter_by_time_control(header, [2, 30], debug)

def filter_by_elo(header: chess.pgn.Headers, white_elo_range, black_elo_range, debug=False):
	logger = logging.getLogger(__name__)
	#headers are often non-standard, try..except!
	discard_game = False
	try:
		if not white_elo_range[0] < header.get("WhiteElo") < white_elo_range[1]:
			discard_game = True
		elif not black_elo_range[0] < header.get("BlackElo") < black_elo_range[1]:
			discard_game = True
	except Exception as e:
		if debug:
			logger.error(f"Exception in filter_by_elo", exec_info=True)
		discard_game = True
	finally:
		return discard_game

def filter_by_time_control(header: chess.pgn.Headers, time_range: List[int], debug: bool = False):
	logger = logging.getLogger(__name__)
	#headers are often non-standard, try..except!
	discard_game = False
	try:
		minutes, increment_seconds = get_time_and_increment_from_time_control_header(header.get("TimeControl"))
		total_time = increment_to_total_time_equivalent(increment_seconds, 40)
		if not time_range[0] < total_time < time_range[1]:
			discard_game = True
	except Exception as e:
		if debug:
			logger.error(f"Exception in filter_by_time_control", exec_info=True)
		discard_game = True
	finally:
		return discard_game
		
def get_time_and_increment_from_time_control_header(time_control: str):
	minute, second = time_control.split("+")
	return int(minute), int(second)

def increment_to_total_time_equivalent(increment: int, moves: int):
	return increment * moves / 60
