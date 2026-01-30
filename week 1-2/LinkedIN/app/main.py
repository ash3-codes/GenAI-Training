# app/main.py
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from rich import print
from rich.console import Console

from memory_store import (
    get_recent_entries,
    load_summary_memory,
    save_summary_memory,
    save_weekly_entry
)
from post_chain import build_post_chain


console = Console()


def calculate_week_id():
    now = datetime.now()
    year, week, _ = now.isocalendar()
    return f"{year}-W{week:02d}"


def format_weekly_memory(entries):
    if not entries:
        return "No past entries available."

    lines = []
    for e in entries:
        lines.append(f"- {e['week']}: topics={e['topics']} | summary={e['summary']}")
    return "\n".join(lines)


def update_summary_memory(old_summary: str, new_topics: list, new_summary: str):
    # summarization 
    if not old_summary:
        return f"Progress story: {', '.join(new_topics)}."

    return old_summary + f" Then moved to {', '.join(new_topics)}."


def main():
    planner_chain, writer_chain, parse_plan = build_post_chain()

    console.print("[bold cyan]LinkedIn Post Generator (Role + Memory + Timeline)[/bold cyan]")

    this_week_text = console.input("\n[bold]What did you do this week?[/bold]\n> ")
    word_limit = console.input("\n[bold]Word limit (e.g., 120)?[/bold]\n> ")

    # memory retrieval
    weekly_entries = get_recent_entries(n=6)
    weekly_memory_text = format_weekly_memory(weekly_entries)
    summary_memory = load_summary_memory()

    # -------- Planner --------
    console.print("\n[bold yellow]Planning...[/bold yellow]")

    plan_text = planner_chain.invoke(
        {
            "weekly_memory": weekly_memory_text,
            "summary_memory": summary_memory,
            "this_week_text": this_week_text,
            "word_limit": word_limit,
        }
    )

    console.print("\n[bold green]Planner Output (JSON):[/bold green]")
    console.print(plan_text)

    plan = parse_plan(plan_text)

    # -------- Writer --------
    console.print("\n[bold yellow]Writing post...[/bold yellow]")

    post = writer_chain.invoke(
        {
            "plan_json": json.dumps(plan, indent=2),
            "word_limit": word_limit,
        }
    )

    console.print("\n[bold magenta]Generated LinkedIn Post:[/bold magenta]\n")
    console.print(post)

    # -------- Save Memory --------
    week_id = calculate_week_id()
    topics = plan.get("this_week_topics", [])

    # create summarized memory entry for future reference
    short_summary = plan.get("story_arc", "") + " " + this_week_text[:150]

    save_weekly_entry(
        week=week_id,
        topics=topics,
        summary=short_summary
    )

    new_summary_memory = update_summary_memory(summary_memory, topics, short_summary)
    save_summary_memory(new_summary_memory)

    console.print("\n[bold cyan]✅ Memory updated successfully.[/bold cyan]")


if __name__ == "__main__":
    main()
