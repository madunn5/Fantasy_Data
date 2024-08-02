import requests
from bs4 import BeautifulSoup
import sqlite3
from datetime import datetime


# Function to create a database table if it doesn't exist
def create_table_if_not_exists(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fantasy_data (
            week INTEGER,
            name TEXT,
            position TEXT,
            team TEXT,
            opponent TEXT,
            actual_points REAL,
            projected_points REAL,
            PRIMARY KEY (week, name)
        )
    ''')
    conn.commit()


# # Connect to the SQLite database
# conn = sqlite3.connect('fantasy_data.db')
#
# # Create the table if it doesn't exist
# create_table_if_not_exists(conn)
#
# # Close the database connection
# conn.close()


# Function to insert data into the database
def insert_data(conn, data):
    cursor = conn.cursor()
    for player in data:
        cursor.execute('''
            INSERT OR REPLACE INTO fantasy_data (week, name, position, team, opponent, actual_points, projected_points)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (week, player['Name'], player['Position'], player['Team'], player['Opponent'], player['Actual Points'],
              player['Projected Points']))
    conn.commit()


# Replace 'YOUR_LEAGUE_ID' with your Yahoo Sports league ID
league_id = '123 Cancun'
url = f"https://football.fantasysports.yahoo.com/f1/{league_id}/players"
week = 1  # Update this with the current week of the season

# Send HTTP GET request to the URL
response = requests.get(url)

# Check if request was successful
if response.status_code == 200:
    # Parse HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table containing player data
    player_table = soup.find('table', class_='Table Table--align-right Table--fixed Table--fixed-left Fz-xs')

    # Extract player data
    players = []
    for row in player_table.find_all('tr')[1:]:
        cells = row.find_all('td')
        player_name = cells[0].text.strip()
        player_position = cells[1].text.strip()
        player_team = cells[2].text.strip()
        # You need to extract opponent, actual points, and projected points from somewhere in the webpage
        player_opponent = cells[...]  # Extract opponent from the webpage
        player_actual_points = cells[...]  # Extract actual points from the webpage
        player_projected_points = cells[...]  # Extract projected points from the webpage
        players.append({
            'Name': player_name,
            'Position': player_position,
            'Team': player_team,
            'Opponent': player_opponent,
            'Actual Points': player_actual_points,
            'Projected Points': player_projected_points
        })

    # Connect to the SQLite database
    conn = sqlite3.connect('fantasy_data.db')

    # Create the table if it doesn't exist
    create_table_if_not_exists(conn)

    # Insert data into the database
    insert_data(conn, players)

    # Close the database connection
    conn.close()

else:
    print("Failed to retrieve data")
