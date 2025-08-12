/*
	Capacitive candle flicker sensing.

	Detects flicker events in an oscillating candle to derive a 
	1Hz frequency for LED blinking.

	cpldcpu July 2025

	Based on the CH32V003 touch sensing example code.

	The mechanism of operation for the touch sensing on the CH32V003 is to:
		* Hold an IO low.
		* Start the ADC
		* Use the internal pull-up to pull the line high.
		* The ADC will sample the voltage on the slope.
		* Lower voltage = longer RC response, so higher capacitance. 
*/

#include "ch32fun.h"
#include <stdio.h>

#include "ch32v003_touch.h"

int main()
{
	SystemInit();

	printf("Capacitive Flicker Sensor\n");
	printf("raw\tavg\thp\tzero_cross\n");
	
	// Enable GPIOD, C and ADC
	RCC->APB2PCENR |= RCC_APB2Periph_GPIOA | RCC_APB2Periph_GPIOD | RCC_APB2Periph_GPIOC | RCC_APB2Periph_ADC1;

	// Configure PC4 as output for LED
	GPIOC->CFGLR &= ~(0xf<<(4*4));
	GPIOC->CFGLR |= (GPIO_Speed_10MHz | GPIO_CNF_OUT_PP)<<(4*4);

	InitTouchADC();
	
	int32_t avg = 0;
	int32_t hp = 0;
	int32_t hp_prev = 0;  // Previous hp value for zero crossing detection
	int32_t zero_cross = 0;  // Zero crossing output
	int32_t dead_time_counter = 0;  // Dead time counter

	// LED blinking variables (using time accumulator for 1Hz LED)

	int32_t led_state = 0;  // LED state (0 = off, 1 = on)
	const int32_t led_toggle_threshold = 32768;  // Toggle LED every 32768 time units (0.5 second)
	const int32_t interval = (int32_t)(65536 / 9.9); // 9.9Hz flicker rate
	int32_t time_accumulator = 0; // Accumulator for time intervals

	while(1)
	{
		uint32_t sum;
		int iterations = 32;
		sum = ReadTouchPin( GPIOA, 2, 0, iterations );
	
		if (avg == 0) { avg = sum;} // initialize avg on first run
		avg = avg - (avg>>5) + sum; // simple low-pass filter
		hp = sum -  (avg>>5); // high-pass filter

		// Zero crossing detector with dead time
		if (dead_time_counter > 0) {
			dead_time_counter--;  // Count down dead time
			zero_cross = 0;  // No detection during dead time
		} else {
			// Check for positive zero crossing (sign change)
			if ((hp_prev < 0 && hp >= 0)) {
				zero_cross = 1;  
				dead_time_counter = 4;  
				time_accumulator += interval;  
				
				// LED blinking logic using time accumulator
				// Check if time accumulator has reached LED toggle threshold
				if (time_accumulator >= led_toggle_threshold) {
					time_accumulator = time_accumulator - led_toggle_threshold;  // Subtract threshold (no modulo)
					led_state = led_state ^ 1;  // Toggle LED state using XOR
					
					// Set or clear PC4 based on LED state
					if (led_state) {
						GPIOC->BSHR = 1<<4;  // Set PC4 high
					} else {
						GPIOC->BSHR = 1<<(16+4);  // Set PC4 low
					}
				}
			} else {
				zero_cross = 0;  // No zero crossing
			}
		}
		
		hp_prev = hp;  // Store current hp for next iteration

		printf( "%d\t%d\t%d\t%d\n", (int)sum, (int)avg, (int)hp, (int)zero_cross );
	}
}

