import asyncio
import threading
import time
from components.speech_to_text import SpeechToTextService
from components.llm_service import LLMService
from components.orchestrator import Orchestrator
from components.rag_service import RAGService
from components.ai_agent import AIAgent
from components.state_manager import StateManager
from components.transcript_storage import TranscriptStorageService
from frontend.gradio_app import GradioInterface
from config import Config


class CustomerSupportAIApp:
    def __init__(self):
        # Initialize storage service
        self.transcript_storage = TranscriptStorageService()

        # Initialize components with storage
        self.state_manager = StateManager(
            transcript_storage=self.transcript_storage
        )
        self.speech_service = SpeechToTextService()
        self.llm_service = LLMService()
        self.rag_service = RAGService()
        self.ai_agent = AIAgent()
        self.orchestrator = Orchestrator(
            self.rag_service, self.ai_agent, self.state_manager
        )
        self.frontend = GradioInterface(self.state_manager)
        self.transcription_task = None
        self.background_loop = None
        self.background_thread = None
        self.is_running = False

    async def initialize(self):
        """Initialize all services"""
        try:
            print("Initializing RAG service...")
            await self.rag_service.initialize()
            print("✓ RAG service initialized")

            print("Initializing AI agent...")
            await self.ai_agent.initialize()
            print("✓ AI agent initialized")

            print("✓ All services initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing services: {e}")
            import traceback

            traceback.print_exc()
            raise

    async def _transcription_loop(self):
        """Background task for transcription using callback approach"""
        print("\n=== Starting transcription loop (callback approach) ===")

        def on_transcript_entry(entry):
            """Callback for when a transcript entry is received"""
            print(
                f"\n🎤 Received entry in callback: {entry.speaker.value}: {entry.text}"
            )

            # Direct approach - add to state manager synchronously
            try:
                print("📝 Adding transcript entry directly...")
                # Since we're in a callback from a different thread, we need to be careful
                # Let's add it directly to the state without async
                with self.state_manager._lock:
                    print(
                        f"📋 Adding transcript entry: {entry.speaker.value}: {entry.text}"
                    )
                    self.state_manager.state.transcript.append(entry)

                    # Auto-save check
                    if (
                        self.transcript_storage
                        and len(self.state_manager.state.transcript) % 5 == 0
                    ):
                        # Schedule auto-save in background loop
                        if self.background_loop:
                            asyncio.run_coroutine_threadsafe(
                                self.state_manager._auto_save_transcript(),
                                self.background_loop,
                            )

                    print("✅ Entry successfully added to state manager")

                    # Verify the entry was added
                    current_count = len(self.state_manager.state.transcript)
                    print(f"📊 Current transcript count: {current_count}")

            except Exception as e:
                print(f"❌ Error adding entry to state manager: {e}")
                import traceback

                traceback.print_exc()

        try:
            # Set up the callback
            print("🔗 Setting up speech callback...")
            self.speech_service.set_entry_callback(on_transcript_entry)

            # Start transcription with callback
            print("🎙️ Starting transcription service...")
            await self.speech_service.start_transcription_with_callback()
        except Exception as e:
            print(f"❌ Error in transcription loop: {e}")
            import traceback

            traceback.print_exc()

    def run_background_loop(self):
        """Run the background async loop in a separate thread"""
        try:
            print("🔄 Creating background event loop...")
            self.background_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.background_loop)

            # Initialize services
            print("🔧 Initializing services in background...")
            self.background_loop.run_until_complete(self.initialize())

            # Start transcription loop
            print("🎤 Starting transcription in background...")
            self.background_loop.run_until_complete(self._transcription_loop())

        except Exception as e:
            print(f"❌ Error in background loop: {e}")
            import traceback

            traceback.print_exc()
        finally:
            if self.background_loop:
                print("🔄 Closing background loop...")
                self.background_loop.close()

    def start_background_services(self):
        """Start background services in a separate thread"""
        self.is_running = True
        print("🚀 Starting background thread...")
        self.background_thread = threading.Thread(
            target=self.run_background_loop, daemon=True
        )
        self.background_thread.start()
        print("✅ Background services started")

        # Give the background thread time to initialize
        print("⏳ Waiting for background services to initialize...")
        time.sleep(3)  # Increased wait time

    def process_trigger_sync(self):
        """Process trigger in sync context for Gradio - now with better formatting"""
        print("\n🤖 === Processing trigger (Generate AI Response) ===")

        # Get current transcript
        current_state = self.state_manager.get_state()
        transcript = current_state.transcript
        print(f"📊 Current transcript length: {len(transcript)}")

        if not transcript:
            print("⚠️ No transcript found")
            return (
                "⚠️ No conversation to process",
                "Please have a conversation first before generating AI assistance.",
            )

        try:
            # Save transcript before processing
            print("💾 Saving transcript before processing...")

            # Create a new event loop for this operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Save current transcript
                saved_file = loop.run_until_complete(
                    self.state_manager.save_current_transcript()
                )
                if saved_file:
                    print(f"✅ Transcript saved to: {saved_file}")

                print("🧠 Generating task from transcript...")
                task = loop.run_until_complete(
                    self.llm_service.generate_task_from_transcript(transcript)
                )
                print(f"📋 Generated task: {task.description}")
                print(f"🏷️ Task type: {task.task_type}")

                print("🚀 Routing task...")
                result = loop.run_until_complete(
                    self.orchestrator.route_task(task)
                )
                print(f"✅ Task result: {result}")

                # Format response for layman operator
                status_info = self._format_task_analysis(task)
                formatted_result = self._format_ai_response(task, result)

                return status_info, formatted_result

            finally:
                loop.close()

        except Exception as e:
            print(f"❌ Error in process_trigger_sync: {str(e)}")
            import traceback

            traceback.print_exc()
            return (
                f"❌ Error: {str(e)}",
                "An error occurred while processing your request. Please try again.",
            )

    def _format_task_analysis(self, task) -> str:
        """Format task analysis in a layman-friendly way"""
        urgency_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

        category_emoji = {
            "Order Status": "📦",
            "Refund Request": "💰",
            "Product Issue": "⚠️",
            "Account Help": "👤",
            "Complaint": "😠",
            "General Inquiry": "❓",
        }

        emoji = urgency_emoji.get(
            getattr(task, "urgency_level", "Medium"), "🟡"
        )
        cat_emoji = category_emoji.get(
            getattr(task, "issue_category", "General Inquiry"), "❓"
        )

        analysis = f"""📋 CUSTOMER SERVICE ANALYSIS
        
{emoji} URGENCY: {getattr(task, 'urgency_level', 'Medium')}
{cat_emoji} CATEGORY: {getattr(task, 'issue_category', 'General Inquiry')}

👤 CUSTOMER: {task.customer_name}
📝 REQUEST: {task.issue_description}"""

        if task.order_number:
            analysis += f"\n🏷️ ORDER: {task.order_number}"

        if hasattr(task, "order_status") and task.order_status:
            analysis += f"\n📊 STATUS: {task.order_status}"

        analysis += f"\n⚙️ ACTION TYPE: {'Information Lookup' if task.task_type == 'rag' else 'Take Action'}"
        analysis += f"\n✅ STATUS: {task.status.title()}"

        # Add verification points if available
        if hasattr(task, "verification_points") and task.verification_points:
            analysis += f"\n\n🔍 PLEASE VERIFY:"
            for point in task.verification_points:
                analysis += f"\n   • {point}"

        return analysis

    def _format_ai_response(self, task, result) -> str:
        """Format AI response in a helpful way for the operator"""
        response = f"""🤖 AI ASSISTANT GUIDANCE

📋 WHAT TO DO:
{getattr(task, 'operator_instructions', 'Review customer request and provide assistance')}

💬 SUGGESTED RESPONSE:
"{getattr(task, 'suggested_response', 'Thank you for contacting us. Let me help you with that.')}"

🔧 SYSTEM RESULT:
{result}

📝 ACTION PLAN:"""

        if task.generated_plan:
            if isinstance(task.generated_plan, list):
                for step in task.generated_plan:
                    response += f"\n{step}"
            else:
                response += f"\n{task.generated_plan}"

        return response

    def stop(self):
        """Stop all services"""
        print("🛑 Stopping all services...")
        self.is_running = False
        if self.speech_service:
            self.speech_service.stop_transcription()
        if self.background_loop and not self.background_loop.is_closed():
            self.background_loop.call_soon_threadsafe(self.background_loop.stop)

    def run(self):
        """Run the application"""
        try:
            # Start background services
            self.start_background_services()

            # Remove the automatic test entry - let users control everything
            print("🎤 Ready for user input - no automatic entries")

            # Create and launch Gradio interface
            interface = self.frontend.create_interface(
                trigger_callback=self.process_trigger_sync
            )

            print("🚀 Launching Gradio interface...")
            interface.launch(
                server_port=Config.GRADIO_PORT, share=False, inbrowser=True
            )

        except Exception as e:
            print(f"❌ Error running application: {e}")
            import traceback

            traceback.print_exc()
            self.stop()
            raise


def main():
    """Entry point"""
    app = CustomerSupportAIApp()

    try:
        app.run()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        app.stop()
    except Exception as e:
        print(f"❌ Application error: {e}")
        app.stop()


if __name__ == "__main__":
    main()
