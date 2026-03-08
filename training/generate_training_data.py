"""
Generate training data to improve tool usage (phi3:mini, qwen2.5-coder:3b, etc.).

Covers these failure patterns:
1. Tool format - correct XML syntax
2. Commands vs tool names - commands go inside <command> tags
3. Code execution vs display - know when to run vs show
4. Stop conditions - when to stop tool calls and answer
5. Tool selection - match intent to correct tool
6. Context focus - stay on topic
"""

import json
import random
from pathlib import Path

# Training examples in chat format
TRAINING_EXAMPLES = [
    # ==========================================================================
    # PATTERN 1: Correct Tool Format
    # ==========================================================================
    {
        "category": "tool_format",
        "conversations": [
            {"role": "user", "content": "List files in current directory"},
            {"role": "assistant", "content": "<tool>bash</tool><command>ls -la</command>"}
        ]
    },
    {
        "category": "tool_format",
        "conversations": [
            {"role": "user", "content": "Run php artisan migrate"},
            {"role": "assistant", "content": "<tool>bash</tool><command>php artisan migrate</command>"}
        ]
    },
    {
        "category": "tool_format",
        "conversations": [
            {"role": "user", "content": "Execute composer install"},
            {"role": "assistant", "content": "<tool>bash</tool><command>composer install</command>"}
        ]
    },
    {
        "category": "tool_format",
        "conversations": [
            {"role": "user", "content": "Search Laravel docs for eloquent"},
            {"role": "assistant", "content": "<tool>laravel_docs</tool><query>eloquent</query>"}
        ]
    },
    {
        "category": "tool_format",
        "conversations": [
            {"role": "user", "content": "Search web for Laravel 11 features"},
            {"role": "assistant", "content": "<tool>web_search</tool><query>Laravel 11 features</query>"}
        ]
    },
    {
        "category": "tool_format",
        "conversations": [
            {"role": "user", "content": "Read the User model file"},
            {"role": "assistant", "content": "<tool>file_ops</tool><action>read</action><path>app/Models/User.php</path>"}
        ]
    },

    # ==========================================================================
    # PATTERN 2: Commands are NOT tool names
    # ==========================================================================
    {
        "category": "command_not_tool",
        "conversations": [
            {"role": "user", "content": "Run ls -la"},
            {"role": "assistant", "content": "<tool>bash</tool><command>ls -la</command>"},
            {"role": "user", "content": "WRONG: [Tool: ls -la] - this is incorrect. ls -la is a command, not a tool name. Always use <tool>bash</tool><command>ls -la</command>"},
            {"role": "assistant", "content": "Understood. Commands like ls, php, composer, npm are executed via the bash tool, not as tool names themselves. Correct format: <tool>bash</tool><command>ls -la</command>"}
        ]
    },
    {
        "category": "command_not_tool",
        "conversations": [
            {"role": "user", "content": "Execute php artisan make:model Post"},
            {"role": "assistant", "content": "<tool>bash</tool><command>php artisan make:model Post</command>"},
            {"role": "user", "content": "Good! You correctly used bash tool with the command inside <command> tags."},
            {"role": "assistant", "content": "Yes, artisan commands are shell commands that run through bash. The tool is 'bash', the command is 'php artisan make:model Post'."}
        ]
    },

    # ==========================================================================
    # PATTERN 3: Code Display vs Execution
    # ==========================================================================
    {
        "category": "code_vs_execute",
        "conversations": [
            {"role": "user", "content": "Show me how to define a route in Laravel"},
            {"role": "assistant", "content": "Here's how to define a route in Laravel:\n\n```php\nRoute::get('/users', [UserController::class, 'index']);\nRoute::post('/users', [UserController::class, 'store']);\n```\n\nThis defines GET and POST routes for the /users endpoint."}
        ]
    },
    {
        "category": "code_vs_execute",
        "conversations": [
            {"role": "user", "content": "Show me a controller example"},
            {"role": "assistant", "content": "Here's a basic Laravel controller:\n\n```php\n<?php\n\nnamespace App\\Http\\Controllers;\n\nuse App\\Models\\User;\nuse Illuminate\\Http\\Request;\n\nclass UserController extends Controller\n{\n    public function index()\n    {\n        return User::all();\n    }\n\n    public function store(Request $request)\n    {\n        return User::create($request->validated());\n    }\n}\n```"}
        ]
    },
    {
        "category": "code_vs_execute",
        "conversations": [
            {"role": "user", "content": "Create a new controller for posts"},
            {"role": "assistant", "content": "I'll create the PostController:\n\n<tool>bash</tool><command>php artisan make:controller PostController --resource</command>"}
        ]
    },
    {
        "category": "code_vs_execute",
        "conversations": [
            {"role": "user", "content": "What does a migration look like?"},
            {"role": "assistant", "content": "Here's an example migration:\n\n```php\n<?php\n\nuse Illuminate\\Database\\Migrations\\Migration;\nuse Illuminate\\Database\\Schema\\Blueprint;\nuse Illuminate\\Support\\Facades\\Schema;\n\nreturn new class extends Migration\n{\n    public function up(): void\n    {\n        Schema::create('posts', function (Blueprint $table) {\n            $table->id();\n            $table->string('title');\n            $table->text('content');\n            $table->foreignId('user_id')->constrained();\n            $table->timestamps();\n        });\n    }\n\n    public function down(): void\n    {\n        Schema::dropIfExists('posts');\n    }\n};\n```"}
        ]
    },
    {
        "category": "code_vs_execute",
        "conversations": [
            {"role": "user", "content": "Run the migration"},
            {"role": "assistant", "content": "<tool>bash</tool><command>php artisan migrate</command>"}
        ]
    },

    # ==========================================================================
    # PATTERN 4: Know When to Stop (Don't over-tool)
    # ==========================================================================
    {
        "category": "stop_condition",
        "conversations": [
            {"role": "user", "content": "What is Laravel?"},
            {"role": "assistant", "content": "Laravel is a PHP web application framework with expressive, elegant syntax. It provides tools for routing, authentication, sessions, caching, and database management. Key features:\n\n- **Eloquent ORM** - Active Record implementation for database\n- **Blade Templates** - Powerful templating engine\n- **Artisan CLI** - Command-line interface for common tasks\n- **Migrations** - Version control for database schema\n- **Built-in Auth** - Authentication scaffolding\n\nIt follows MVC architecture and emphasizes clean, maintainable code."}
        ]
    },
    {
        "category": "stop_condition",
        "conversations": [
            {"role": "user", "content": "What is Eloquent?"},
            {"role": "assistant", "content": "Eloquent is Laravel's built-in ORM (Object-Relational Mapping). It lets you interact with your database using PHP objects instead of raw SQL.\n\nKey features:\n- Each table has a corresponding Model class\n- Relationships: hasMany, belongsTo, hasOne, belongsToMany\n- Query scopes for reusable query logic\n- Automatic timestamps (created_at, updated_at)\n- Soft deletes, mutators, accessors\n\nExample:\n```php\n$users = User::where('active', true)->get();\n$user->posts()->create(['title' => 'Hello']);\n```"}
        ]
    },
    {
        "category": "stop_condition",
        "conversations": [
            {"role": "user", "content": "Explain middleware"},
            {"role": "assistant", "content": "Middleware in Laravel filters HTTP requests entering your application.\n\nCommon uses:\n- **Authentication** - Check if user is logged in\n- **CORS** - Handle cross-origin requests\n- **Logging** - Log request/response data\n- **Rate Limiting** - Throttle requests\n\nExample:\n```php\nclass CheckAge\n{\n    public function handle($request, Closure $next)\n    {\n        if ($request->age < 18) {\n            return redirect('home');\n        }\n        return $next($request);\n    }\n}\n```\n\nApply to routes:\n```php\nRoute::get('/admin', fn() => ...)->middleware('auth');\n```"}
        ]
    },

    # ==========================================================================
    # PATTERN 5: Correct Tool Selection
    # ==========================================================================
    {
        "category": "tool_selection",
        "conversations": [
            {"role": "user", "content": "Check what files are in app/Models"},
            {"role": "assistant", "content": "<tool>file_ops</tool><action>list</action><path>app/Models</path>"}
        ]
    },
    {
        "category": "tool_selection",
        "conversations": [
            {"role": "user", "content": "Find information about Laravel queues"},
            {"role": "assistant", "content": "<tool>laravel_docs</tool><query>queues jobs</query>"}
        ]
    },
    {
        "category": "tool_selection",
        "conversations": [
            {"role": "user", "content": "What are the latest Laravel packages for authentication?"},
            {"role": "assistant", "content": "<tool>web_search</tool><query>Laravel authentication packages 2024</query>"}
        ]
    },
    {
        "category": "tool_selection",
        "conversations": [
            {"role": "user", "content": "Read the routes file"},
            {"role": "assistant", "content": "<tool>file_ops</tool><action>read</action><path>routes/web.php</path>"}
        ]
    },
    {
        "category": "tool_selection",
        "conversations": [
            {"role": "user", "content": "Install a new package"},
            {"role": "assistant", "content": "Which package would you like to install? For example:\n\n<tool>bash</tool><command>composer require laravel/sanctum</command>"}
        ]
    },

    # ==========================================================================
    # PATTERN 6: Context Focus (Stay on Topic)
    # ==========================================================================
    {
        "category": "context_focus",
        "conversations": [
            {"role": "user", "content": "How do I create a migration for a posts table?"},
            {"role": "assistant", "content": "To create a migration for posts:\n\n<tool>bash</tool><command>php artisan make:migration create_posts_table</command>\n\nThis creates a migration file. Add your schema:\n\n```php\nSchema::create('posts', function (Blueprint $table) {\n    $table->id();\n    $table->string('title');\n    $table->text('content');\n    $table->foreignId('user_id')->constrained();\n    $table->timestamps();\n});\n```\n\nThen run: `php artisan migrate`"}
        ]
    },
    {
        "category": "context_focus",
        "conversations": [
            {"role": "user", "content": "I'm building a blog, what models do I need?"},
            {"role": "assistant", "content": "For a blog, you'll typically need these models:\n\n1. **User** - Authors\n2. **Post** - Blog posts\n3. **Category** - Post categories\n4. **Tag** - Post tags (many-to-many)\n5. **Comment** - User comments\n\nRelationships:\n- User hasMany Posts\n- Post belongsTo User\n- Post belongsToMany Tags\n- Post belongsTo Category\n- Post hasMany Comments\n\nCreate them:\n```bash\nphp artisan make:model Post -m\nphp artisan make:model Category -m\nphp artisan make:model Tag -m\nphp artisan make:model Comment -m\n```"}
        ]
    },

    # ==========================================================================
    # PATTERN 7: NEVER Execute PHP Code via Bash (CRITICAL)
    # ==========================================================================
    {
        "category": "no_execute_php",
        "conversations": [
            {"role": "user", "content": "Show me a model example"},
            {"role": "assistant", "content": "Here's a Laravel model example:\n\n```php\n<?php\n\nnamespace App\\Models;\n\nuse Illuminate\\Database\\Eloquent\\Model;\n\nclass Post extends Model\n{\n    protected $fillable = ['title', 'content'];\n\n    public function user()\n    {\n        return $this->belongsTo(User::class);\n    }\n}\n```\n\nThis model has fillable fields and a relationship to User."}
        ]
    },
    {
        "category": "no_execute_php",
        "conversations": [
            {"role": "user", "content": "Give me validation rules example"},
            {"role": "assistant", "content": "Here's how to define validation rules:\n\n```php\n$rules = [\n    'title' => 'required|string|max:255',\n    'email' => 'required|email|unique:users',\n    'password' => 'required|min:8|confirmed',\n];\n```\n\nOr in a FormRequest:\n\n```php\npublic function rules()\n{\n    return [\n        'title' => 'required|max:255',\n        'body' => 'required',\n    ];\n}\n```"}
        ]
    },
    {
        "category": "no_execute_php",
        "conversations": [
            {"role": "user", "content": "How do I query with Eloquent?"},
            {"role": "assistant", "content": "Here are common Eloquent queries:\n\n```php\n// Get all records\n$users = User::all();\n\n// Find by ID\n$user = User::find(1);\n\n// Where clauses\n$active = User::where('active', true)->get();\n\n// First or fail\n$user = User::findOrFail($id);\n\n// Create\n$post = Post::create(['title' => 'Hello']);\n\n// Update\n$user->update(['name' => 'New Name']);\n```"}
        ]
    },

    # ==========================================================================
    # PATTERN 8: More Bash Format Examples
    # ==========================================================================
    {
        "category": "bash_format",
        "conversations": [
            {"role": "user", "content": "List all files"},
            {"role": "assistant", "content": "<tool>bash</tool><command>ls -la</command>"}
        ]
    },
    {
        "category": "bash_format",
        "conversations": [
            {"role": "user", "content": "Show current directory"},
            {"role": "assistant", "content": "<tool>bash</tool><command>pwd</command>"}
        ]
    },
    {
        "category": "bash_format",
        "conversations": [
            {"role": "user", "content": "Check PHP version"},
            {"role": "assistant", "content": "<tool>bash</tool><command>php -v</command>"}
        ]
    },
    {
        "category": "bash_format",
        "conversations": [
            {"role": "user", "content": "Run tests"},
            {"role": "assistant", "content": "<tool>bash</tool><command>php artisan test</command>"}
        ]
    },
    {
        "category": "bash_format",
        "conversations": [
            {"role": "user", "content": "Clear cache"},
            {"role": "assistant", "content": "<tool>bash</tool><command>php artisan cache:clear</command>"}
        ]
    },
    {
        "category": "bash_format",
        "conversations": [
            {"role": "user", "content": "Install npm packages"},
            {"role": "assistant", "content": "<tool>bash</tool><command>npm install</command>"}
        ]
    },

    # ==========================================================================
    # PATTERN 9: Direct Answers (No Tools Needed)
    # ==========================================================================
    {
        "category": "direct_answer",
        "conversations": [
            {"role": "user", "content": "What is MVC?"},
            {"role": "assistant", "content": "MVC stands for Model-View-Controller, a design pattern that separates application logic:\n\n- **Model**: Data and business logic (Eloquent models)\n- **View**: User interface (Blade templates)\n- **Controller**: Handles requests, coordinates Model and View\n\nLaravel follows MVC architecture to organize code cleanly."}
        ]
    },
    {
        "category": "direct_answer",
        "conversations": [
            {"role": "user", "content": "What is a facade?"},
            {"role": "assistant", "content": "Facades in Laravel provide a static interface to classes in the service container. They offer a simple syntax for common operations.\n\nExamples:\n- `Cache::get('key')` - Access cache\n- `DB::table('users')` - Database queries\n- `Route::get()` - Define routes\n- `Auth::user()` - Get authenticated user\n\nFacades are syntactic sugar over dependency injection."}
        ]
    },
    {
        "category": "direct_answer",
        "conversations": [
            {"role": "user", "content": "Explain service providers"},
            {"role": "assistant", "content": "Service providers are the central place to configure your Laravel application. They:\n\n1. **Register bindings** in the service container\n2. **Bootstrap services** like routes, events, middleware\n\nKey methods:\n- `register()`: Bind things into container\n- `boot()`: Called after all providers registered\n\nLocation: `app/Providers/`\n\nCore providers: AppServiceProvider, RouteServiceProvider, AuthServiceProvider."}
        ]
    },

    # ==========================================================================
    # PATTERN 10: Multi-turn Context
    # ==========================================================================
    {
        "category": "multi_turn",
        "conversations": [
            {"role": "user", "content": "I want to add authentication to my app"},
            {"role": "assistant", "content": "Laravel has several authentication options:\n\n1. **Laravel Breeze** - Simple, minimal auth scaffolding\n2. **Laravel Jetstream** - Full-featured with teams, 2FA\n3. **Laravel Fortify** - Backend only, no views\n4. **Laravel Sanctum** - API token authentication\n\nFor a typical web app, I recommend Breeze. Should I show you how to install it?"},
            {"role": "user", "content": "Yes, install Breeze"},
            {"role": "assistant", "content": "<tool>bash</tool><command>composer require laravel/breeze --dev</command>\n\nAfter installation, run:\n```bash\nphp artisan breeze:install\nnpm install && npm run dev\nphp artisan migrate\n```\n\nThis gives you login, registration, password reset, and email verification."}
        ]
    },
    {
        "category": "multi_turn",
        "conversations": [
            {"role": "user", "content": "Create a Post model"},
            {"role": "assistant", "content": "<tool>bash</tool><command>php artisan make:model Post -mf</command>\n\nThis creates:\n- `app/Models/Post.php` - Model\n- `database/migrations/xxx_create_posts_table.php` - Migration\n- `database/factories/PostFactory.php` - Factory"},
            {"role": "user", "content": "Now add a relationship to User"},
            {"role": "assistant", "content": "Add this to your Post model (`app/Models/Post.php`):\n\n```php\npublic function user()\n{\n    return $this->belongsTo(User::class);\n}\n```\n\nAnd in your User model:\n\n```php\npublic function posts()\n{\n    return $this->hasMany(Post::class);\n}\n```\n\nDon't forget to add `user_id` to your posts migration:\n```php\n$table->foreignId('user_id')->constrained()->cascadeOnDelete();\n```"}
        ]
    },
]


def generate_training_file(output_path: str = "training_data.jsonl"):
    """Generate training data in JSONL format."""
    output_file = Path(output_path)

    with open(output_file, 'w') as f:
        for example in TRAINING_EXAMPLES:
            # Format for fine-tuning
            entry = {
                "messages": [
                    {"role": "system", "content": "You are a Laravel AI Assistant. Use tools with exact XML format: <tool>name</tool><param>value</param>"},
                    *example["conversations"]
                ],
                "category": example["category"]
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Generated {len(TRAINING_EXAMPLES)} training examples to {output_file}")
    return output_file


def generate_ollama_modelfile(base_model: str = "qwen2.5-coder:3b", output_path: str = "Modelfile"):
    """Generate Ollama Modelfile for fine-tuning."""

    # Collect all conversations as few-shot examples
    few_shot = []
    for example in TRAINING_EXAMPLES[:15]:  # Take first 15 examples
        for msg in example["conversations"]:
            few_shot.append(f"{msg['role'].upper()}: {msg['content']}")

    few_shot_text = "\n\n".join(few_shot)

    modelfile = f'''FROM {base_model}

SYSTEM """You are a Laravel AI Assistant. ALWAYS use tools with exact XML format.

TOOLS:
- bash: <tool>bash</tool><command>your command</command>
- web_search: <tool>web_search</tool><query>search query</query>
- laravel_docs: <tool>laravel_docs</tool><query>docs query</query>
- file_ops: <tool>file_ops</tool><action>read|list</action><path>path</path>

CRITICAL RULES:
1. Commands like ls, php, composer go INSIDE <command> tags, NOT as tool names
2. WRONG: [Tool: ls -la]
3. RIGHT: <tool>bash</tool><command>ls -la</command>
4. For showing code examples, just show the code - don't execute it
5. For running commands, use the bash tool
6. Answer questions directly without unnecessary tool calls

EXAMPLES:
{few_shot_text}
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_predict 512
'''

    with open(output_path, 'w') as f:
        f.write(modelfile)

    print(f"Generated Modelfile at {output_path}")
    print(f"To create custom model (Zima will use it if installed): ollama create coding-assistant -f {output_path}")
    return output_path


def generate_stats():
    """Show training data statistics."""
    categories = {}
    for example in TRAINING_EXAMPLES:
        cat = example["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print("\nTraining Data Statistics:")
    print("=" * 40)
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} examples")
    print(f"  TOTAL: {len(TRAINING_EXAMPLES)} examples")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate training data for Laravel Assistant")
    parser.add_argument("--output", "-o", default="training_data.jsonl", help="Output file")
    parser.add_argument("--modelfile", "-m", action="store_true", help="Generate Ollama Modelfile")
    parser.add_argument("--stats", "-s", action="store_true", help="Show statistics")
    parser.add_argument("--base-model", default="qwen2.5-coder:3b", help="Base model for Modelfile")

    args = parser.parse_args()

    if args.stats:
        generate_stats()
    elif args.modelfile:
        generate_ollama_modelfile(args.base_model)
    else:
        generate_training_file(args.output)
        generate_stats()
