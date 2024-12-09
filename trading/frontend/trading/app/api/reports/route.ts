import { promises as fs } from 'fs';
import path from 'path';

export async function GET() {
    try {
        // 读取reports目录
        const reportsDir = path.join(process.cwd(), 'public/reports');
        const files = await fs.readdir(reportsDir);

        // 读取所有JSON文件
        const reports = await Promise.all(
            files
                .filter(file => file.endsWith('.json'))
                .map(async file => {
                    const content = await fs.readFile(
                        path.join(reportsDir, file),
                        'utf-8'
                    );
                    return JSON.parse(content);
                })
        );

        return Response.json(reports);
    } catch (error) {
        console.error('Error loading reports:', error);
        return Response.json({ error: 'Failed to load reports' }, { status: 500 });
    }
} 