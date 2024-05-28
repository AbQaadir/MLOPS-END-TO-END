import random

class SuccessJourney:
    def __init__(self):
        self.effort = 0
        self.persistent_attempts = 0
        self.failures = 0
        self.success = False
    
    def put_in_effort(self):
        self.effort += 1
        print(f"💪 Putting in effort: {self.effort}")
    
    def attempt_success(self):
        if random.random() < (self.effort / 10):  # Higher effort increases chance of success
            self.success = True
        else:
            self.failures += 1
            self.persistent_attempts += 1
            self.learn_from_failure()
            print("💔 Failure encountered. Learning and trying again...")

    def learn_from_failure(self):
        # Each failure teaches something, so effort increases
        self.effort += 1
        print(f"📚 Learning from failure. Effort increased to: {self.effort}")
        print("🌱 Each failure is a step towards growth. Keep pushing!")

    def journey_towards_success(self):
        print("🌟 Starting the journey towards success!")
        while not self.success:
            self.put_in_effort()
            self.attempt_success()
        
        print(f"🎉 Success achieved after {self.persistent_attempts} persistent attempts and {self.failures} failures!")
        print(f"🚀 Total effort put in: {self.effort}")
        print("🌟 Remember, every step, every failure, and every effort brought you here. Keep believing in yourself!")

# Create an instance of SuccessJourney and start the journey
journey = SuccessJourney()
journey.journey_towards_success()
