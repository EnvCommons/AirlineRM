from openreward.environments import Server

from airlinerm import AirlineRM

if __name__ == "__main__":
    server = Server([AirlineRM])
    server.run()
