"""
Laravel Documentation Tool - RAG-based Laravel knowledge retrieval
"""

import os
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class DocChunk:
    title: str
    content: str
    source: str
    score: float = 0.0


class LaravelDocsTool:
    """Search Laravel documentation using semantic search (RAG)."""

    name = "laravel_docs"
    description = "Search Laravel documentation for information about Laravel features, best practices, and code examples."

    # Core Laravel documentation topics
    LARAVEL_KNOWLEDGE = {
        "routing": {
            "title": "Laravel Routing",
            "content": """Laravel routing defines how your application responds to HTTP requests.

Basic Routes:
```php
Route::get('/users', [UserController::class, 'index']);
Route::post('/users', [UserController::class, 'store']);
Route::put('/users/{id}', [UserController::class, 'update']);
Route::delete('/users/{id}', [UserController::class, 'destroy']);
```

Route Parameters:
```php
Route::get('/users/{id}', function ($id) {
    return User::findOrFail($id);
});
```

Route Groups:
```php
Route::middleware(['auth'])->group(function () {
    Route::get('/dashboard', [DashboardController::class, 'index']);
});

Route::prefix('admin')->group(function () {
    Route::get('/users', [AdminUserController::class, 'index']);
});
```

Named Routes:
```php
Route::get('/profile', [ProfileController::class, 'show'])->name('profile');
// Generate URL: route('profile')
```

Resource Routes:
```php
Route::resource('posts', PostController::class);
// Creates: index, create, store, show, edit, update, destroy
```"""
        },
        "controllers": {
            "title": "Laravel Controllers",
            "content": """Controllers handle HTTP requests and return responses.

Create Controller:
```bash
php artisan make:controller UserController
php artisan make:controller UserController --resource
php artisan make:controller UserController --api
```

Basic Controller:
```php
namespace App\\Http\\Controllers;

use App\\Models\\User;
use Illuminate\\Http\\Request;

class UserController extends Controller
{
    public function index()
    {
        $users = User::paginate(15);
        return view('users.index', compact('users'));
    }

    public function store(Request $request)
    {
        $validated = $request->validate([
            'name' => 'required|string|max:255',
            'email' => 'required|email|unique:users',
        ]);

        $user = User::create($validated);
        return redirect()->route('users.show', $user);
    }

    public function show(User $user)
    {
        return view('users.show', compact('user'));
    }
}
```

API Controller:
```php
class ApiUserController extends Controller
{
    public function index()
    {
        return User::all();
    }

    public function store(Request $request)
    {
        $user = User::create($request->validated());
        return response()->json($user, 201);
    }
}
```"""
        },
        "models": {
            "title": "Laravel Eloquent Models",
            "content": """Eloquent ORM provides an elegant ActiveRecord implementation.

Create Model:
```bash
php artisan make:model Post
php artisan make:model Post -m  # with migration
php artisan make:model Post -mfc  # with migration, factory, controller
```

Model Definition:
```php
namespace App\\Models;

use Illuminate\\Database\\Eloquent\\Model;
use Illuminate\\Database\\Eloquent\\Factories\\HasFactory;
use Illuminate\\Database\\Eloquent\\SoftDeletes;

class Post extends Model
{
    use HasFactory, SoftDeletes;

    protected $fillable = ['title', 'content', 'user_id'];

    protected $casts = [
        'published_at' => 'datetime',
        'is_featured' => 'boolean',
    ];

    // Relationships
    public function user()
    {
        return $this->belongsTo(User::class);
    }

    public function comments()
    {
        return $this->hasMany(Comment::class);
    }

    public function tags()
    {
        return $this->belongsToMany(Tag::class);
    }

    // Scopes
    public function scopePublished($query)
    {
        return $query->whereNotNull('published_at');
    }

    // Accessors
    public function getTitleAttribute($value)
    {
        return ucfirst($value);
    }
}
```

Query Examples:
```php
$posts = Post::where('user_id', 1)->get();
$post = Post::findOrFail($id);
$posts = Post::with('user', 'comments')->paginate(10);
$post = Post::firstOrCreate(['slug' => $slug], $data);
```"""
        },
        "migrations": {
            "title": "Laravel Migrations",
            "content": """Migrations are version control for your database schema.

Create Migration:
```bash
php artisan make:migration create_posts_table
php artisan make:migration add_status_to_posts_table
```

Migration Example:
```php
use Illuminate\\Database\\Migrations\\Migration;
use Illuminate\\Database\\Schema\\Blueprint;
use Illuminate\\Support\\Facades\\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('posts', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->cascadeOnDelete();
            $table->string('title');
            $table->string('slug')->unique();
            $table->text('content');
            $table->enum('status', ['draft', 'published', 'archived'])->default('draft');
            $table->timestamp('published_at')->nullable();
            $table->timestamps();
            $table->softDeletes();

            $table->index(['status', 'published_at']);
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('posts');
    }
};
```

Run Migrations:
```bash
php artisan migrate
php artisan migrate:rollback
php artisan migrate:fresh --seed
```"""
        },
        "validation": {
            "title": "Laravel Validation",
            "content": """Laravel provides powerful validation for incoming data.

Controller Validation:
```php
public function store(Request $request)
{
    $validated = $request->validate([
        'title' => 'required|string|max:255',
        'email' => 'required|email|unique:users,email',
        'password' => 'required|min:8|confirmed',
        'age' => 'nullable|integer|min:18',
        'tags' => 'array',
        'tags.*' => 'string|max:50',
    ]);

    User::create($validated);
}
```

Form Request:
```bash
php artisan make:request StoreUserRequest
```

```php
class StoreUserRequest extends FormRequest
{
    public function authorize(): bool
    {
        return true;
    }

    public function rules(): array
    {
        return [
            'name' => 'required|string|max:255',
            'email' => 'required|email|unique:users',
            'password' => 'required|min:8|confirmed',
        ];
    }

    public function messages(): array
    {
        return [
            'email.unique' => 'This email is already registered.',
        ];
    }
}
```

Custom Validation Rule:
```php
use Illuminate\\Validation\\Rule;

'email' => [
    'required',
    Rule::unique('users')->ignore($user->id),
],
```"""
        },
        "authentication": {
            "title": "Laravel Authentication",
            "content": """Laravel provides built-in authentication scaffolding.

Setup with Breeze:
```bash
composer require laravel/breeze --dev
php artisan breeze:install
npm install && npm run dev
php artisan migrate
```

Manual Authentication:
```php
use Illuminate\\Support\\Facades\\Auth;

// Attempt login
if (Auth::attempt(['email' => $email, 'password' => $password])) {
    $request->session()->regenerate();
    return redirect()->intended('dashboard');
}

// Get authenticated user
$user = Auth::user();
$id = Auth::id();

// Check if authenticated
if (Auth::check()) {
    // User is logged in
}

// Logout
Auth::logout();
$request->session()->invalidate();
$request->session()->regenerateToken();
```

Middleware:
```php
Route::middleware(['auth'])->group(function () {
    Route::get('/dashboard', [DashboardController::class, 'index']);
});

// In controller
public function __construct()
{
    $this->middleware('auth');
}
```

API Authentication (Sanctum):
```bash
composer require laravel/sanctum
php artisan vendor:publish --provider="Laravel\\Sanctum\\SanctumServiceProvider"
```

```php
// Create token
$token = $user->createToken('api-token')->plainTextToken;

// Protect routes
Route::middleware('auth:sanctum')->get('/user', function (Request $request) {
    return $request->user();
});
```"""
        },
        "middleware": {
            "title": "Laravel Middleware",
            "content": """Middleware filters HTTP requests entering your application.

Create Middleware:
```bash
php artisan make:middleware EnsureUserIsAdmin
```

```php
namespace App\\Http\\Middleware;

use Closure;
use Illuminate\\Http\\Request;

class EnsureUserIsAdmin
{
    public function handle(Request $request, Closure $next)
    {
        if (! $request->user()?->isAdmin()) {
            abort(403, 'Unauthorized');
        }

        return $next($request);
    }
}
```

Register Middleware (bootstrap/app.php in Laravel 11):
```php
->withMiddleware(function (Middleware $middleware) {
    $middleware->alias([
        'admin' => \\App\\Http\\Middleware\\EnsureUserIsAdmin::class,
    ]);
})
```

Use Middleware:
```php
Route::get('/admin', [AdminController::class, 'index'])->middleware('admin');

Route::middleware(['auth', 'admin'])->group(function () {
    // Admin routes
});
```"""
        },
        "blade": {
            "title": "Laravel Blade Templates",
            "content": """Blade is Laravel's powerful templating engine.

Basic Syntax:
```blade
{{-- Echoing data (escaped) --}}
{{ $user->name }}

{{-- Unescaped output --}}
{!! $html !!}

{{-- Comments --}}
{{-- This is a comment --}}

{{-- Conditionals --}}
@if($user->isAdmin())
    <p>Welcome, Admin!</p>
@elseif($user->isModerator())
    <p>Welcome, Moderator!</p>
@else
    <p>Welcome, User!</p>
@endif

@unless($user->isBanned())
    <p>You can post</p>
@endunless

{{-- Loops --}}
@foreach($users as $user)
    <p>{{ $loop->iteration }}. {{ $user->name }}</p>
@endforeach

@forelse($posts as $post)
    <p>{{ $post->title }}</p>
@empty
    <p>No posts found.</p>
@endforelse
```

Layouts:
```blade
{{-- layouts/app.blade.php --}}
<!DOCTYPE html>
<html>
<head>
    <title>@yield('title', 'My App')</title>
</head>
<body>
    @include('partials.nav')

    <main>
        @yield('content')
    </main>
</body>
</html>

{{-- pages/home.blade.php --}}
@extends('layouts.app')

@section('title', 'Home')

@section('content')
    <h1>Welcome</h1>
@endsection
```

Components:
```blade
{{-- components/alert.blade.php --}}
@props(['type' => 'info', 'message'])

<div class="alert alert-{{ $type }}">
    {{ $message }}
</div>

{{-- Usage --}}
<x-alert type="success" message="Saved!" />
```"""
        },
        "livewire": {
            "title": "Laravel Livewire",
            "content": """Livewire is a full-stack framework for building dynamic interfaces.

Installation:
```bash
composer require livewire/livewire
```

Create Component:
```bash
php artisan make:livewire Counter
```

Component Class:
```php
namespace App\\Livewire;

use Livewire\\Component;

class Counter extends Component
{
    public int $count = 0;

    public function increment()
    {
        $this->count++;
    }

    public function decrement()
    {
        $this->count--;
    }

    public function render()
    {
        return view('livewire.counter');
    }
}
```

Component View:
```blade
<div>
    <h1>{{ $count }}</h1>
    <button wire:click="increment">+</button>
    <button wire:click="decrement">-</button>
</div>
```

Include in Page:
```blade
<livewire:counter />
{{-- or --}}
@livewire('counter')
```

Form Example:
```php
class CreatePost extends Component
{
    public string $title = '';
    public string $content = '';

    protected $rules = [
        'title' => 'required|min:3',
        'content' => 'required|min:10',
    ];

    public function save()
    {
        $validated = $this->validate();
        Post::create($validated);
        $this->reset();
        session()->flash('message', 'Post created!');
    }
}
```

```blade
<form wire:submit="save">
    <input type="text" wire:model="title">
    @error('title') <span>{{ $message }}</span> @enderror

    <textarea wire:model="content"></textarea>
    @error('content') <span>{{ $message }}</span> @enderror

    <button type="submit">Save</button>
</form>
```"""
        },
        "queues": {
            "title": "Laravel Queues",
            "content": """Queues allow deferring time-consuming tasks for background processing.

Create Job:
```bash
php artisan make:job ProcessPodcast
```

Job Class:
```php
namespace App\\Jobs;

use Illuminate\\Bus\\Queueable;
use Illuminate\\Contracts\\Queue\\ShouldQueue;
use Illuminate\\Foundation\\Bus\\Dispatchable;
use Illuminate\\Queue\\InteractsWithQueue;
use Illuminate\\Queue\\SerializesModels;

class ProcessPodcast implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    public function __construct(
        public Podcast $podcast
    ) {}

    public function handle(): void
    {
        // Process the podcast...
    }

    public function failed(\\Throwable $exception): void
    {
        // Handle failure...
    }
}
```

Dispatch Jobs:
```php
ProcessPodcast::dispatch($podcast);
ProcessPodcast::dispatch($podcast)->onQueue('podcasts');
ProcessPodcast::dispatch($podcast)->delay(now()->addMinutes(10));
```

Run Queue Worker:
```bash
php artisan queue:work
php artisan queue:work --queue=high,default
php artisan queue:work --tries=3
```

Configure Queue (config/queue.php):
```php
'default' => env('QUEUE_CONNECTION', 'database'),

'connections' => [
    'database' => [
        'driver' => 'database',
        'table' => 'jobs',
        'queue' => 'default',
        'retry_after' => 90,
    ],
    'redis' => [
        'driver' => 'redis',
        'connection' => 'default',
        'queue' => 'default',
    ],
],
```"""
        },
        "events": {
            "title": "Laravel Events",
            "content": """Events provide a simple observer implementation for decoupled code.

Create Event and Listener:
```bash
php artisan make:event OrderShipped
php artisan make:listener SendShipmentNotification --event=OrderShipped
```

Event Class:
```php
namespace App\\Events;

use Illuminate\\Foundation\\Events\\Dispatchable;
use Illuminate\\Queue\\SerializesModels;

class OrderShipped
{
    use Dispatchable, SerializesModels;

    public function __construct(
        public Order $order
    ) {}
}
```

Listener Class:
```php
namespace App\\Listeners;

use App\\Events\\OrderShipped;

class SendShipmentNotification
{
    public function handle(OrderShipped $event): void
    {
        // Access order via $event->order
        Mail::to($event->order->user)->send(new OrderShippedMail($event->order));
    }
}
```

Register in EventServiceProvider:
```php
protected $listen = [
    OrderShipped::class => [
        SendShipmentNotification::class,
    ],
];
```

Dispatch Event:
```php
event(new OrderShipped($order));
// or
OrderShipped::dispatch($order);
```"""
        },
        "testing": {
            "title": "Laravel Testing",
            "content": """Laravel is built with testing in mind using PHPUnit and Pest.

Create Tests:
```bash
php artisan make:test UserTest
php artisan make:test UserTest --unit
```

Feature Test:
```php
namespace Tests\\Feature;

use Tests\\TestCase;
use App\\Models\\User;
use Illuminate\\Foundation\\Testing\\RefreshDatabase;

class UserTest extends TestCase
{
    use RefreshDatabase;

    public function test_users_can_view_dashboard(): void
    {
        $user = User::factory()->create();

        $response = $this->actingAs($user)
            ->get('/dashboard');

        $response->assertStatus(200);
    }

    public function test_users_can_create_posts(): void
    {
        $user = User::factory()->create();

        $response = $this->actingAs($user)
            ->post('/posts', [
                'title' => 'Test Post',
                'content' => 'Test content',
            ]);

        $response->assertRedirect('/posts');
        $this->assertDatabaseHas('posts', ['title' => 'Test Post']);
    }
}
```

Unit Test:
```php
namespace Tests\\Unit;

use PHPUnit\\Framework\\TestCase;
use App\\Services\\Calculator;

class CalculatorTest extends TestCase
{
    public function test_addition(): void
    {
        $calculator = new Calculator();
        $this->assertEquals(4, $calculator->add(2, 2));
    }
}
```

Run Tests:
```bash
php artisan test
php artisan test --filter=UserTest
php artisan test --parallel
```"""
        }
    }

    def __init__(self, knowledge_dir: Optional[str] = None):
        self.knowledge_dir = Path(knowledge_dir) if knowledge_dir else None
        self.embeddings = None
        self.index = None
        self._chunks: list[DocChunk] = []

    def _load_embeddings(self):
        """Lazy-load sentence transformers for embeddings."""
        if self.embeddings is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                return False
        return True

    def search(self, query: str, top_k: int = 3) -> list[DocChunk]:
        """
        Search Laravel documentation.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant documentation chunks
        """
        query_lower = query.lower()
        results = []

        # Simple keyword matching for built-in knowledge
        for key, doc in self.LARAVEL_KNOWLEDGE.items():
            score = 0

            # Check title match
            if key in query_lower or any(word in query_lower for word in doc['title'].lower().split()):
                score += 5

            # Check content keywords
            content_lower = doc['content'].lower()
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 2 and word in content_lower:
                    score += 1

            if score > 0:
                results.append(DocChunk(
                    title=doc['title'],
                    content=doc['content'],
                    source=f"laravel-docs/{key}",
                    score=score
                ))

        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def run(self, query: str, top_k: int = 3) -> str:
        """
        Search and return formatted documentation.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            Formatted documentation string
        """
        results = self.search(query, top_k)

        if not results:
            return f"No documentation found for: {query}"

        output = []
        for i, doc in enumerate(results, 1):
            output.append(f"## {i}. {doc.title}")
            output.append(f"Source: {doc.source}")
            output.append("")
            output.append(doc.content)
            output.append("")
            output.append("---")
            output.append("")

        return "\n".join(output)

    def get_topic(self, topic: str) -> Optional[str]:
        """Get documentation for a specific topic."""
        topic_lower = topic.lower().strip()

        if topic_lower in self.LARAVEL_KNOWLEDGE:
            doc = self.LARAVEL_KNOWLEDGE[topic_lower]
            return f"# {doc['title']}\n\n{doc['content']}"

        # Try partial match
        for key, doc in self.LARAVEL_KNOWLEDGE.items():
            if topic_lower in key or key in topic_lower:
                return f"# {doc['title']}\n\n{doc['content']}"

        return None

    def list_topics(self) -> list[str]:
        """List all available documentation topics."""
        return list(self.LARAVEL_KNOWLEDGE.keys())

    def to_schema(self) -> dict:
        """Return tool schema for LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for Laravel documentation"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 3)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }


if __name__ == "__main__":
    # Test the tool
    tool = LaravelDocsTool()

    print("=== Available Topics ===")
    print(tool.list_topics())

    print("\n=== Search: 'create controller' ===")
    print(tool.run("create controller"))

    print("\n=== Search: 'authentication login' ===")
    print(tool.run("authentication login"))
