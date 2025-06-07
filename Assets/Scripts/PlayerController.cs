using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.InputSystem;

public class PlayerController : MonoBehaviour
{
 // Rigidbody of the player.
 private Rigidbody rb; 

 // Movement along X and Y axes.
 private float movementX;
 private float movementY;

 // Current velocity of the player
 private Vector3 currentVelocity;

 // Movement parameters
 public float maxSpeed = 10f;        // Maximum speed the player can reach
 public float acceleration = 5f;     // How quickly the player accelerates
 public float deceleration = 10f;     // How quickly the player decelerates

 // Start is called before the first frame update.
 void Start()
    {
 // Get and store the Rigidbody component attached to the player.
        rb = GetComponent<Rigidbody>();
        currentVelocity = Vector3.zero;
    }
 
 // This function is called when a move input is detected.
 void OnMove(InputValue movementValue)
    {
 // Convert the input value into a Vector2 for movement.
        Vector2 movementVector = movementValue.Get<Vector2>();

 // Store the X and Y components of the movement.
        movementX = movementVector.x; 
        movementY = movementVector.y; 
    }

 // FixedUpdate is called once per fixed frame-rate frame.
 private void FixedUpdate() 
    {
 // Create a 3D movement vector using the X and Y inputs.
        Vector3 targetDirection = new Vector3(movementX, 0.0f, movementY);
        
        if (targetDirection.magnitude > 0)
        {
            // Normalize the direction to ensure consistent acceleration
            targetDirection.Normalize();
            
            // Accelerate towards the target direction
            currentVelocity = Vector3.MoveTowards(
                currentVelocity,
                targetDirection * maxSpeed,
                acceleration * Time.fixedDeltaTime
            );
        }
        else
        {
            // Decelerate when no input is detected
            currentVelocity = Vector3.MoveTowards(
                currentVelocity,
                Vector3.zero,
                deceleration * Time.fixedDeltaTime
            );
        }

        // Apply the calculated velocity to the rigidbody
        rb.linearVelocity = currentVelocity;
    }
}