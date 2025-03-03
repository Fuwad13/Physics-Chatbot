import json

# Define your data as a list of dictionaries
dataset = [
    {
        "question": "How are rest and motion defined in physics?",
        "ideal_answer": "An object is said to be at rest when its position does not change with time in relation to its surroundings. An object is in motion when its position changes with time relative to its surroundings."
    },
    {
        "question": "What is the definition of work in physics?",
        "ideal_answer": "Work is done when a force is applied to an object, and the object moves in the direction of the applied force. It is calculated as the product of force and displacement in the direction of the force."
    },
    {
        "question": "What is thermal expansion, and how does it affect solids, liquids, and gases?",
        "ideal_answer": "Thermal expansion refers to the increase in the size of a substance when its temperature increases. Solids expand the least, liquids expand more, and gases expand the most when heated."
    },
    {
        "question": "What is a wave, and how is sound produced?",
        "ideal_answer": "A wave is a disturbance that transfers energy through a medium without transferring matter. Sound is produced by vibrating objects, and it travels as a longitudinal wave through a medium like air, water, or solids."
    },
    {
        "question": "What is refraction, and why does it occur?",
        "ideal_answer": "Refraction is the bending of light as it passes from one medium to another with a different density. It occurs because light changes speed when moving between materials with different refractive indices."
    }
]

# Save to a JSON file
with open("dataset.json", "w") as file:
    json.dump(dataset, file, indent=4)



