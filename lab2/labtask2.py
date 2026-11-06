
class ClassroomAgent:

    def __init__(self):
        self.previous_mode = None

    def act(self, temp, occupied):

        if not occupied:
            target_mode = "ECO"

        elif temp > 26:
            target_mode = "COOL"
        elif temp < 20:
            target_mode = "WARM"
        else:
            target_mode = "IDLE"
        if target_mode != self.previous_mode:

            send_command = True
            self.previous_mode = target_mode
        else:
            send_command = False
        return target_mode, send_command

agent = ClassroomAgent()
percepts = [
    (29, True),
    (29, True),
    (20, True),
    (26, True),
    (19, True),
    (29, True),
    (29, False),
    (29, False)
]
print("SMART CLASSROOM AGENT")
print("----------------------------------------------")
for step, (temp, occupied) in enumerate(percepts, 1):
    previous = agent.previous_mode
    target, command = agent.act(temp, occupied)
    if command:
        result = "SEND COMMAND"
    else:
        result = "NO COMMAND"
    print("Step:", step)
    print("Previous Mode:", previous)
    print("Temperature:", temp)
    print("Occupied:", occupied)
    print("Target Mode:", target)
    print("Command:", result)
    print("----------------------------------------------")
