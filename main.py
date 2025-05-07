import sys
import json
from collections import deque, defaultdict


def is_operator(c: str) -> bool:
    return c in {'|', '*', '(', ')', '.', '+', '?'}

def adauga_concatenare(pattern: str) -> str:
    result = []
    for c in pattern:
        if c == ' ':
            continue
        if result:
            prev = result[-1]
            lit_before = not is_operator(prev) or prev in {')', '*', '+', '?'}
            lit_now = not is_operator(c) or c == '('
            if lit_before and lit_now:
                result.append('.')
        result.append(c)
    return ''.join(result)

def priority(op: str) -> int:
    return {'|': 1, '.': 2, '*': 3, '+': 3, '?': 3}.get(op, 0)

def to_postfix(infix_expr: str) -> str:
    expr = adauga_concatenare(infix_expr)
    output, stack = [], []
    for c in expr:
        if c == '(':
            stack.append(c)
        elif c == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if not stack:
                raise ValueError('Paranteze neinchise corect')
            stack.pop()
        elif not is_operator(c):
            output.append(c)
        elif c in {'*', '+', '?'}:
            output.append(c)
        else:
            while stack and priority(stack[-1]) >= priority(c):
                output.append(stack.pop())
            stack.append(c)
    while stack:
        if stack[-1] == '(':
            raise ValueError('Paranteze neinchise corect')
        output.append(stack.pop())
    return ''.join(output)


class StateNFA:
    def __init__(self):
        self.transitions = defaultdict(set)
        self.lambda_moves = set()

class Fragment:
    def __init__(self, start: int, accept: int):
        self.start = start
        self.accept = accept

class NFA:
    def __init__(self):
        self.states = []

    def add_state(self) -> int:
        self.states.append(StateNFA())
        return len(self.states) - 1

    @staticmethod
    def build_from_postfix(postfix: str):
        nfa = NFA()
        stack = []

        def literal(ch: str):
            s = nfa.add_state()
            t = nfa.add_state()
            nfa.states[s].transitions[ch].add(t)
            stack.append(Fragment(s, t))

        for token in postfix:
            if token == '*':
                frag = stack.pop()
                s = nfa.add_state()
                t = nfa.add_state()
                nfa.states[s].lambda_moves |= {frag.start, t}
                nfa.states[frag.accept].lambda_moves |= {frag.start, t}
                stack.append(Fragment(s, t))
            elif token == '+':
                frag = stack.pop()
                s = nfa.add_state()
                t = nfa.add_state()
                nfa.states[s].lambda_moves |= {frag.start}
                nfa.states[frag.accept].lambda_moves |= {frag.start, t}
                stack.append(Fragment(s, t))
            elif token == '?':
                frag = stack.pop()
                s = nfa.add_state()
                t = nfa.add_state()
                nfa.states[s].lambda_moves |= {frag.start, t}
                nfa.states[frag.accept].lambda_moves.add(t)
                stack.append(Fragment(s, t))
            elif token == '|':
                right = stack.pop()
                left = stack.pop()
                s = nfa.add_state()
                t = nfa.add_state()
                nfa.states[s].lambda_moves |= {left.start, right.start}
                nfa.states[left.accept].lambda_moves.add(t)
                nfa.states[right.accept].lambda_moves.add(t)
                stack.append(Fragment(s, t))
            elif token == '.':
                right = stack.pop()
                left = stack.pop()
                nfa.states[left.accept].lambda_moves.add(right.start)
                stack.append(Fragment(left.start, right.accept))
            else:
                literal(token)

        if len(stack) != 1:
            raise ValueError('Regex postfix invalid')
        frag = stack[0]
        return nfa, frag.start, frag.accept


class DFA:
    def __init__(self):
        self.transitions = []
        self.final_states = []
        self.start = 0

    def match(self, text: str) -> bool:
        state = self.start
        for c in text:
            state = self.transitions[state].get(c)
            if state is None:
                return False
        return self.final_states[state]


def lambda_closure(nfa: NFA, state_set: set) -> set:
    stack = list(state_set)
    closure = set(state_set)
    while stack:
        u = stack.pop()
        for v in nfa.states[u].lambda_moves:
            if v not in closure:
                closure.add(v)
                stack.append(v)
    return closure


def convert_nfa_to_dfa(nfa: NFA, start: int, accept: int) -> DFA:
    dfa = DFA()
    start_set = frozenset(lambda_closure(nfa, {start}))
    state_ids = {start_set: 0}
    dfa.transitions.append({})
    dfa.final_states.append(accept in start_set)
    queue = deque([start_set])
    alphabet = {sym for st in nfa.states for sym in st.transitions}

    while queue:
        current = queue.popleft()
        current_id = state_ids[current]
        for sym in alphabet:
            move_set = set()
            for u in current:
                move_set |= nfa.states[u].transitions.get(sym, set())
            if not move_set:
                continue
            next_set = frozenset(lambda_closure(nfa, move_set))
            if next_set not in state_ids:
                state_ids[next_set] = len(dfa.transitions)
                dfa.transitions.append({})
                dfa.final_states.append(accept in next_set)
                queue.append(next_set)
            dfa.transitions[current_id][sym] = state_ids[next_set]
    return dfa


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'tests.json'
    try:
        test_cases = json.load(open(path, encoding='utf-8'))
    except FileNotFoundError:
        print(f"Fisierul {path} nu a fost gasit", file=sys.stderr)
        sys.exit(1)

    total = passed = 0
    for test in test_cases:
        name = test.get('name', '')
        postfix = to_postfix(test['regex'])
        nfa, start, accept = NFA.build_from_postfix(postfix)
        dfa = convert_nfa_to_dfa(nfa, start, accept)
        for t in test['test_strings']:
            total += 1
            expected = t['expected']
            result = dfa.match(t['input'])
            if result == expected:
                passed += 1
                print(f"{name}: OK ({t['input']})")
            else:
                print(f"{name}: FAIL ({t['input']}) expected={expected} got={result}")
    print(f"{passed}/{total} correct")


if __name__ == '__main__':
    main()