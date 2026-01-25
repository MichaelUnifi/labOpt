import pycutest

names = pycutest.find_problems(objective='other', constraints='bound', regular=True, userN=True)

problems = []
for name in names:
    p = pycutest.import_problem(name)
    problems.append((name, p.n))

# Sort by number of variables
problems.sort(key=lambda x: x[1])

for name, nvar in problems:
    print("Name: ", name, "Number of variables: ", nvar)