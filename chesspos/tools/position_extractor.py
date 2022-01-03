import argparse

from chesspos.preprocessing.pgn_extract import pgn_to_encoding

if __name__ == "__main__":

	# https://stackoverflow.com/questions/29335145/python-argparse-extra-args/29335524#29335524
	def ensure_value(namespace, dest, default):
		stored = getattr(namespace, dest, None)
		if stored is None:
			return default
		return stored

	class store_dict(argparse.Action):
		def __call__(self, parser, namespace, values, option_string=None):
			vals = dict(ensure_value(namespace, self.dest, {}))
			k, _, v = values.partition('=')
			vals[k] = v
			setattr(namespace, self.dest, vals)

	parser = argparse.ArgumentParser(description='Generate bitboards and training samples')

	parser.add_argument('input', type=str, action="store", help='pgn file with input games')
	parser.add_argument('--profile', type=bool, action="store", default=False, help='profile the code')
	parser.add_argument(
		'--format', type=str, default="bitboard", action="store",
		help='encoding format: bitboard(default)|tensor'
	)
	parser.add_argument(
		'--save_position', type=str, action="store",
		help='h5py file to store the encoded positions'
	)
	parser.add_argument(
		'--tuples', type=bool, default=False, action="store",
		help='Do you want to generate tuples of positions: True|False(default)'
	)
	parser.add_argument(
		'--save_tuples', type=str, action="store",
		help='h5py file to store the encoded tuples'
	)
	parser.add_argument(
		'--chunksize', type=int, action="store", default=100000,
		help='Chunk size for paginating games'
	)
	parser.add_argument(
		'--filter', default={}, action=store_dict,
		help="Filter out games. Options: time_min, elo_min. Usage: --filter key1=val1 --filter key2=val2",
		metavar="KEY1=VAL1"
	)

	args = parser.parse_args()

	print(f"Input file at: {args.input}")
	print(f"Encoding format: {args.format}")
	print(f"Filter options: {args.filter}")
	print(f"Positions saved at: {args.save_position}")
	print(f"Tuples generated: {args.tuples}")
	print(f"Tuples saved at: {args.save_tuples}")
	print(f"Chunksize: {args.chunksize}\n\n")

	if args.profile:
		from pyinstrument import Profiler

		profiler = Profiler() # or Profiler(use_signal=False), see below
		profiler.start()

		pgn_to_encoding(
			args.input,
			format=args.format,
			save_file=args.save_position,
			generate_tuples=args.tuples,
			tuple_file=args.save_tuples,
			chunksize=args.chunksize,
			game_filter=args.filter
		)

		profiler.stop()
		print(profiler.output_text(unicode=True, color=True))

	else:
		pgn_to_encoding(
			args.input,
			format=args.format,
			save_file=args.save_position,
			generate_tuples=args.tuples,
			tuple_file=args.save_tuples,
			chunksize=args.chunksize,
			game_filter=args.filter
		)
