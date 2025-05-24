from utils.models import Task
import asyncio
from datetime import datetime


class AIAgent:
    """Simple AI Agent without browser automation for testing"""

    def __init__(self):
        print("\n=== Initializing Simple AIAgent (no browser) ===")
        self.start_time = None

    async def initialize(self):
        """Initialize simple agent"""
        self.start_time = datetime.now()
        print(f"\n=== Simple agent initialized at {self.start_time} ===")
        print("✓ Simple AI agent ready (browser automation disabled)")

    async def execute_task(self, task: Task) -> str:
        """Execute task using simple simulation"""
        task_start_time = datetime.now()
        print(f"\n=== Starting task at {task_start_time} ===")
        print(f"Task: {task.description}")

        try:
            # Simulate processing time
            await asyncio.sleep(2)

            # Simple task simulation based on description
            description_lower = task.description.lower()

            if "refund" in description_lower:
                if task.order_number:
                    result = f"✅ Refund processed for order {task.order_number}. Amount: $12.00. Refund will appear in 3-5 business days."
                else:
                    result = "✅ Refund request processed. Customer will receive confirmation email shortly."
            elif "cancel" in description_lower:
                if task.order_number:
                    result = f"✅ Order {task.order_number} has been cancelled. Refund will be processed automatically."
                else:
                    result = "✅ Order cancellation processed. Customer will receive confirmation email."
            elif "order" in description_lower and "status" in description_lower:
                if task.order_number:
                    result = f"📦 Order {task.order_number} is currently being processed and will ship within 2-3 business days."
                else:
                    result = "📦 Order status checked. Customer will receive tracking information via email."
            else:
                result = f"✅ Customer request processed: {task.description}. Support team will follow up if needed."

            task_end_time = datetime.now()
            duration = (task_end_time - task_start_time).total_seconds()
            print(f"✓ Task completed in {duration:.2f} seconds")
            print(f"Result: {result}")

            return result

        except Exception as e:
            print(f"❌ Error executing task: {str(e)}")
            return f"Error executing task: {str(e)}"

    async def cleanup(self):
        """Cleanup resources"""
        print("\n=== Cleaning up simple agent ===")
        if self.start_time:
            total_duration = (datetime.now() - self.start_time).total_seconds()
            print(f"Total runtime: {total_duration:.2f} seconds")
        print("✓ Simple agent cleanup completed")
