from argparse import ArgumentParser

# so it looks for -f and -b flags

parser = ArgumentParser(description='Description of your program')
parser.add_argument('-f','--foo', help='Description for foo argument')#, required=True)
parser.add_argument('-b','--bar', help='Description for bar argument')#, required=True)

# if you want it to be a dictionary
args = vars(parser.parse_args()) 

if args['foo'] == 'Hello':
    print('dic',args['foo'])
else:
    print('dic not hello')

if args['bar'] == 'World':
    print('dic',args['bar'])
else:
    print('dic not world')

# if you want it as a namespace object
args = parser.parse_args()

if args.foo == 'Hello':
    print('object',args.foo)
else:
    print('object not hello')



# This one you have to put them in an order, I'd rather have them named
# import sys

# print('Number of arguments:', len(sys.argv), 'arguments.')
# print('Argument List:', str(sys.argv))

# print('hello world')